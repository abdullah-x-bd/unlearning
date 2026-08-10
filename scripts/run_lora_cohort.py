from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.lora import attach_lora, save_cohort_adapter
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.plan import build_plan
from unlearning_at_scale.state import model_sha256
from unlearning_at_scale.training import TraceRunner, create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and delete a cohort-scoped LoRA adapter over a frozen base")
    parser.add_argument("config")
    parser.add_argument("--cohort-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    model_cfg = config["model"]
    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    base, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    base.to(device)
    base_hash_before = model_sha256(base)
    cohort_ids = [line.strip() for line in Path(args.cohort_ids).read_text().splitlines() if line.strip()]
    store = TokenStore(config["data"]["train_dir"], dummy_token_id=tokenizer.pad_token_id)
    cohort_ids = [sample_id for sample_id in cohort_ids if sample_id in store.id_to_row]
    if not cohort_ids:
        raise SystemExit("No cohort IDs occur in the prepared dataset")

    peft_model = attach_lora(base, rank=args.rank, alpha=args.alpha, dropout=0.0)
    opt_cfg = config["optimizer"]
    optimizer = create_optimizer(
        peft_model,
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        foreach=bool(opt_cfg.get("foreach", False)),
        fused=bool(opt_cfg.get("fused", False)),
    )
    plan_cfg = config["plan"]
    plan = build_plan(
        cohort_ids,
        microbatch_size=int(plan_cfg["microbatch_size"]),
        grad_accum_steps=int(plan_cfg["grad_accum_steps"]),
        epochs=int(plan_cfg.get("epochs", 1)),
        shuffle_seed=int(plan_cfg.get("shuffle_seed", seed)),
        rng_seed=int(plan_cfg.get("rng_seed", seed + 1)),
        peak_lr=float(opt_cfg["lr"]),
        warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.0)),
        schedule=opt_cfg.get("schedule", "constant"),
        shuffle=bool(plan_cfg.get("shuffle", True)),
    )
    stats = TraceRunner(peft_model, optimizer, store, device=device, dtype=model_cfg.get("dtype", "fp32")).run(plan)
    output = Path(args.output)
    adapter_dir = output / "adapter"
    save_cohort_adapter(peft_model, adapter_dir)
    adapter_bytes = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())

    recovered_base = peft_model.unload()
    recovered_hash = model_sha256(recovered_base)
    payload = {
        "cohort_examples": len(cohort_ids),
        "training": stats.to_dict(),
        "base_sha256_before_adapter": base_hash_before,
        "base_sha256_after_unload": recovered_hash,
        "exact_base_recovery": base_hash_before == recovered_hash,
        "adapter_bytes": adapter_bytes,
        "rank": args.rank,
        "alpha": args.alpha,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
