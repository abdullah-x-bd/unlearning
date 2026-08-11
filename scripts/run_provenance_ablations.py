from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unlearning_at_scale.ablations import ablate_plan, supported_ablations
from unlearning_at_scale.compare import compare_state_dicts
from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.plan import read_plan
from unlearning_at_scale.state import load_checkpoint, model_sha256, optimizer_sha256
from unlearning_at_scale.training import TraceRunner, create_optimizer


def new_model_optimizer(config: dict, device: torch.device):
    model_cfg = config["model"]
    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    model.to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--run-dir", required=True, help="Completed original run containing execution_plan.jsonl and original checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ablations", nargs="*", default=list(supported_ablations()))
    args = parser.parse_args()

    config = load_config(args.config)
    configure_determinism(int(config.get("seed", 2026)), strict=bool(config.get("strict_determinism", True)))
    device = torch.device(config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_plan(run_dir / "execution_plan.jsonl")
    base = run_dir / "original" / "checkpoints" / "step-000000.pt"
    reference_state = torch.load(run_dir / "original" / "final-model-state.pt", map_location="cpu", weights_only=True)
    store = TokenStore(config["data"]["train_dir"])

    rows = []
    for name in args.ablations:
        model, tokenizer, optimizer = new_model_optimizer(config, device)
        load_checkpoint(base, model, optimizer, map_location=device)
        altered = ablate_plan(plan, name)
        stats = TraceRunner(model, optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32")).run(altered)
        comp = compare_state_dicts(reference_state, model.state_dict()).to_dict()
        rows.append({
            "ablation": name,
            "stats": stats.to_dict(),
            "model_sha256": model_sha256(model),
            "optimizer_sha256": optimizer_sha256(optimizer),
            "comparison_to_original": comp,
        })
        del model, tokenizer, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output / "provenance_ablations.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
