from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unlearning_at_scale.compare import compare_state_dicts
from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.state import load_checkpoint, model_sha256, optimizer_sha256
from unlearning_at_scale.training import TraceRunner, create_optimizer
from unlearning_at_scale.wal import WalReader


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def new_model_optimizer(config: dict):
    model_cfg = config["model"]
    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    opt_cfg = config["optimizer"]
    optimizer = create_optimizer(
        model,
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        foreach=bool(opt_cfg.get("foreach", False)),
        fused=bool(opt_cfg.get("fused", False)),
    )
    return model, tokenizer, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure checkpoint storage versus replay latency from a completed run")
    parser.add_argument("config")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    model_cfg = config["model"]
    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir)
    scenario_dir = run_dir / "forget" / args.scenario
    forget_ids = {line.strip() for line in (scenario_dir / "forget_ids.txt").read_text().splitlines() if line.strip()}
    scenario_summary = json.loads((scenario_dir / "summary.json").read_text())
    earliest = int(scenario_summary["earliest_affected_step"])
    oracle_state = torch.load(scenario_dir / "oracle" / "final-model-state.pt", map_location="cpu", weights_only=True)
    oracle_summary = json.loads((scenario_dir / "oracle" / "summary.json").read_text())
    oracle_hash = oracle_summary["model_sha256"]

    checkpoint_dir = run_dir / "original" / "checkpoints"
    all_checkpoints = sorted(checkpoint_dir.glob("step-*.pt"), key=checkpoint_step)
    if not all_checkpoints:
        raise SystemExit("No checkpoints found")

    reconstructed = WalReader.from_environment(run_dir / "original" / "trace.wal", run_dir / "original" / "manifest.jsonl").to_plan()
    store = TokenStore(config["data"]["train_dir"])
    redacted_store = store.redact(forget_ids)
    rows = []

    for interval in sorted(set(args.intervals)):
        if interval <= 0:
            raise ValueError("checkpoint intervals must be positive")
        retained = [path for path in all_checkpoints if checkpoint_step(path) == 0 or checkpoint_step(path) % interval == 0]
        eligible = [path for path in retained if checkpoint_step(path) <= earliest]
        if not eligible:
            continue
        chosen = eligible[-1]
        chosen_step = checkpoint_step(chosen)
        model, tokenizer, optimizer = new_model_optimizer(config)
        load_checkpoint(chosen, model, optimizer, map_location=device)
        stats = TraceRunner(model, optimizer, redacted_store, device=device, dtype=model_cfg.get("dtype", "fp32")).run(
            reconstructed,
            forget_ids=forget_ids,
            policy="slot_mask",
            start_optimizer_step=chosen_step,
        )
        comp = compare_state_dicts(oracle_state, model.state_dict()).to_dict()
        rows.append({
            "checkpoint_interval_steps": interval,
            "chosen_checkpoint_step": chosen_step,
            "replay_distance_steps": max(0, max(spec.optimizer_step for spec in reconstructed) - chosen_step + 1),
            "retained_checkpoint_count": len(retained),
            "retained_checkpoint_bytes": sum(path.stat().st_size for path in retained),
            "chosen_checkpoint_bytes": chosen.stat().st_size,
            "replay_wall_seconds": stats.wall_seconds,
            "model_sha256": model_sha256(model),
            "optimizer_sha256": optimizer_sha256(optimizer),
            "model_hash_equal_to_oracle": model_sha256(model) == oracle_hash,
            "comparison_to_oracle": comp,
        })
        del model, tokenizer, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
