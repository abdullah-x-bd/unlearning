from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism, environment_snapshot
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.plan import build_plan
from unlearning_at_scale.training import TraceRunner, create_optimizer


def tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Pythia 2.8B release memory and throughput")
    parser.add_argument("--config", default="configs/pythia-2.8b-scaling.yaml")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--output", default="results/pythia-2.8b-calibration.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload: dict = {
        "status": "started",
        "requested_updates": args.updates,
    }

    try:
        verify_lock(args.sources, args.lock)
        config = yaml.safe_load(Path(args.config).read_text())
        configure_determinism(int(config.get("seed", 2026)), strict=bool(config.get("strict_determinism", True)))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Pythia 2.8B calibration")

        model_cfg = config["model"]
        revision = revision_from_lock(args.lock, model_cfg["artifact_key"])
        load_started = time.perf_counter()
        model, tokenizer = load_causal_lm(
            model_cfg["name"],
            revision=revision,
            attention_implementation=model_cfg.get("attention_implementation", "eager"),
            disable_dropout=bool(model_cfg.get("disable_dropout", False)),
        )
        device = torch.device(model_cfg.get("device", "cuda"))
        model.to(device)
        model_load_seconds = time.perf_counter() - load_started

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
        needed_examples = args.updates * examples_per_update
        if needed_examples > len(store.ids):
            raise ValueError(f"Calibration requests {needed_examples} examples but store has {len(store.ids)}")

        calibration_ids = store.ids[:needed_examples]
        plan = build_plan(
            calibration_ids,
            microbatch_size=int(plan_cfg["microbatch_size"]),
            grad_accum_steps=int(plan_cfg["grad_accum_steps"]),
            epochs=1,
            shuffle_seed=int(plan_cfg.get("shuffle_seed", config.get("seed", 2026))),
            rng_seed=int(plan_cfg.get("rng_seed", int(config.get("seed", 2026)) + 1)),
            peak_lr=float(opt_cfg["lr"]),
            warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.0)),
            schedule=opt_cfg.get("schedule", "constant"),
            shuffle=True,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        runner = TraceRunner(
            model,
            optimizer,
            store,
            device=device,
            dtype=model_cfg.get("dtype", "fp32"),
        )
        stats = runner.run(
            plan,
            progress_every=max(1, min(10, args.updates)),
            progress_label="pythia-2.8b-calibration",
        )
        torch.cuda.synchronize(device)

        full_updates = len(store.ids) // examples_per_update
        if len(store.ids) % examples_per_update:
            full_updates += 1
        seconds_per_update = stats.wall_seconds / max(1, stats.applied_updates + stats.skipped_updates)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        dtype_counts: dict[str, int] = {}
        for parameter in model.parameters():
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()

        payload = {
            "status": "passed",
            "model": model_cfg["name"],
            "revision": revision,
            "requested_updates": args.updates,
            "examples_per_update": examples_per_update,
            "model_load_seconds": model_load_seconds,
            "calibration": stats.to_dict(),
            "seconds_per_update": seconds_per_update,
            "full_dataset_updates": full_updates,
            "estimated_seconds_per_full_pass_from_calibration": seconds_per_update * full_updates,
            "estimated_seconds_for_six_full_pass_equivalents": seconds_per_update * full_updates * 6,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
            "parameter_dtype_counts": dtype_counts,
            "optimizer_state_tensor_bytes_after_calibration": tensor_bytes(optimizer.state_dict()),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "cuda_free_bytes_after_calibration": free_bytes,
            "cuda_total_bytes": total_bytes,
            "environment": environment_snapshot(),
            "wall_seconds_total_including_model_load": time.perf_counter() - started,
        }
    except Exception as exc:
        payload = {
            **payload,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds_total": time.perf_counter() - started,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        raise

    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
