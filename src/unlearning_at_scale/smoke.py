from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .compare import compare_state_dicts
from .dataset import TokenStore
from .determinism import configure_determinism
from .plan import build_plan, write_plan
from .state import load_checkpoint, save_checkpoint
from .training import TraceRunner, create_optimizer
from .wal import WalReader, WalWriter


class MiniCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 12):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.proj = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def _write_fixture(directory: Path) -> TokenStore:
    directory.mkdir(parents=True, exist_ok=True)
    ids = [f"sample-{i:03d}" for i in range(12)]
    rng = np.random.default_rng(99)
    tokens = rng.integers(1, 31, size=(len(ids), 10), dtype=np.int64)
    mask = np.ones_like(tokens)
    (directory / "ids.json").write_text(json.dumps(ids))
    np.save(directory / "input_ids.npy", tokens)
    np.save(directory / "attention_mask.npy", mask)
    return TokenStore(directory)


def run_core_smoke(output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure_determinism(7, strict=True)
    store = _write_fixture(output / "fixture")
    plan = build_plan(
        store.ids,
        microbatch_size=3,
        grad_accum_steps=2,
        epochs=1,
        shuffle_seed=11,
        rng_seed=13,
        peak_lr=0.01,
        schedule="constant",
    )
    write_plan(output / "plan.jsonl", plan)

    model = MiniCausalLM()
    optimizer = create_optimizer(model, lr=0.01)
    base = output / "base.pt"
    save_checkpoint(base, model, optimizer, 0)

    writer = WalWriter(output / "trace.wal", output / "manifest.jsonl")
    original = TraceRunner(model, optimizer, store, device="cpu")
    original.run(plan, wal_writer=writer)
    writer.close()

    forget = {plan[1].sample_ids[0], plan[2].sample_ids[1]}
    reconstructed = WalReader(output / "trace.wal", output / "manifest.jsonl").to_plan()

    oracle_model = MiniCausalLM()
    oracle_opt = create_optimizer(oracle_model, lr=0.01)
    load_checkpoint(base, oracle_model, oracle_opt)
    TraceRunner(oracle_model, oracle_opt, store, device="cpu").run(plan, forget, "slot_mask")

    replay_model = MiniCausalLM()
    replay_opt = create_optimizer(replay_model, lr=0.01)
    load_checkpoint(base, replay_model, replay_opt)
    TraceRunner(replay_model, replay_opt, store, device="cpu").run(reconstructed, forget, "slot_mask")

    comparison = compare_state_dicts(oracle_model.state_dict(), replay_model.state_dict()).to_dict()
    payload = {"exact": comparison["exact"], "comparison": comparison, "wal_records": len(reconstructed)}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
