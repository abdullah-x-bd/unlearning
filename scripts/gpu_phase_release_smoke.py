from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism, resolve_cuda_index
from unlearning_at_scale.experiment import _release
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.plan import build_plan
from unlearning_at_scale.training import TraceRunner, create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a completed 2.8B phase releases CUDA model and optimizer state")
    parser.add_argument("--config", default="configs/pythia-2.8b-scaling.yaml")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--output", default="results/gpu-phase-release-smoke.json")
    args = parser.parse_args()

    verify_lock(args.sources, args.lock)
    config = yaml.safe_load(Path(args.config).read_text())
    configure_determinism(int(config.get("seed", 2026)), strict=bool(config.get("strict_determinism", True)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the phase-release smoke test")

    model_cfg = config["model"]
    revision = revision_from_lock(args.lock, model_cfg["artifact_key"])
    device = torch.device(model_cfg.get("device", "cuda"))
    cuda_index = resolve_cuda_index(device)

    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=revision,
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
    store = TokenStore(config["data"]["train_dir"], dummy_token_id=tokenizer.pad_token_id)
    plan_cfg = config["plan"]
    examples_per_update = int(plan_cfg["microbatch_size"]) * int(plan_cfg["grad_accum_steps"])
    plan = build_plan(
        store.ids[:examples_per_update],
        microbatch_size=int(plan_cfg["microbatch_size"]),
        grad_accum_steps=int(plan_cfg["grad_accum_steps"]),
        epochs=1,
        shuffle_seed=int(plan_cfg.get("shuffle_seed", config.get("seed", 2026))),
        rng_seed=int(plan_cfg.get("rng_seed", int(config.get("seed", 2026)) + 1)),
        peak_lr=float(opt_cfg["lr"]),
        warmup_ratio=0.0,
        schedule="constant",
        shuffle=False,
    )
    runner = TraceRunner(model, optimizer, store, device=device, dtype=model_cfg.get("dtype", "fp32"))
    stats = runner.run(plan, progress_every=1, progress_label="phase-release-smoke")
    torch.cuda.synchronize(cuda_index)
    allocated_before_release = torch.cuda.memory_allocated(cuda_index)
    peak_before_release = torch.cuda.max_memory_allocated(cuda_index)

    _release(runner, optimizer, model, tokenizer)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cuda_index)
    allocated_after_release = torch.cuda.memory_allocated(cuda_index)
    if allocated_after_release > 512 * 1024 * 1024:
        raise RuntimeError(
            f"phase release left {allocated_after_release / (1024 ** 3):.2f} GiB allocated on CUDA"
        )

    second_model, second_tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=revision,
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    second_model.to(device)
    torch.cuda.synchronize(cuda_index)
    allocated_after_second_load = torch.cuda.memory_allocated(cuda_index)
    _release(second_model, second_tokenizer)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cuda_index)
    allocated_after_second_release = torch.cuda.memory_allocated(cuda_index)
    if allocated_after_second_release > 512 * 1024 * 1024:
        raise RuntimeError(
            f"second phase release left {allocated_after_second_release / (1024 ** 3):.2f} GiB allocated on CUDA"
        )

    payload = {
        "status": "passed",
        "model": model_cfg["name"],
        "revision": revision,
        "one_update_stats": stats.to_dict(),
        "allocated_before_release_bytes": allocated_before_release,
        "peak_before_release_bytes": peak_before_release,
        "allocated_after_release_bytes": allocated_after_release,
        "allocated_after_second_load_bytes": allocated_after_second_load,
        "allocated_after_second_release_bytes": allocated_after_second_release,
        "release_threshold_bytes": 512 * 1024 * 1024,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
