from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.plan import MicrobatchSpec
from unlearning_at_scale.training import TraceRunner, create_optimizer


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.projection(hidden))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_batch_smoke")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        input_ids = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=np.int64,
        )
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        labels = input_ids.copy()
        np.save(directory / "input_ids.npy", input_ids)
        np.save(directory / "attention_mask.npy", attention_mask)
        np.save(directory / "labels.npy", labels)
        (directory / "ids.json").write_text(json.dumps(["a", "b"]))

        store = TokenStore(directory)
        model = TinyCausalLM()
        optimizer = create_optimizer(model, lr=1e-4)
        runner = TraceRunner(model, optimizer, store, device="cuda", dtype="bf16")
        plan = [
            MicrobatchSpec(
                index=0,
                sample_ids=("a", "b"),
                seed=2026,
                lr=1e-4,
                optimizer_step=0,
                accumulation_end=True,
            )
        ]
        stats = runner.run(plan)

        if stats.applied_updates != 1:
            raise RuntimeError(f"Expected one optimizer update, got {stats.applied_updates}")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError("Non-finite parameter detected after CUDA smoke update")

        print(json.dumps({"cuda_batch_smoke": "passed", "stats": stats.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
