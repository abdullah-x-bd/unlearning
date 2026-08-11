from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.config import config_sha256
from unlearning_at_scale.dataset import TokenStore, materialize_redacted_store
from unlearning_at_scale.determinism import configure_determinism, environment_snapshot
from unlearning_at_scale.experiment import _new_model_optimizer
from unlearning_at_scale.forget import select_scenario_forget_ids
from unlearning_at_scale.lifecycle import release_phase
from unlearning_at_scale.plan import build_plan, read_plan, write_plan
from unlearning_at_scale.state import load_checkpoint, model_sha256, optimizer_sha256, save_checkpoint
from unlearning_at_scale.training import TraceRunner
from unlearning_at_scale.wal import WalReader, WalWriter


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_hf(model, tokenizer, output: Path, *, role: str, model_hash: str, optimizer_hash: str, base_revision: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    save_json(
        output / "UAS_SOURCE.json",
        {
            "role": role,
            "model_sha256": model_hash,
            "optimizer_sha256": optimizer_hash,
            "base_revision": base_revision,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the frozen TOFU original and physically-redacted slot replay states for OpenUnlearning evaluation"
    )
    parser.add_argument("--config", default="configs/benchmarks/tofu-llama32-1b-forget10.yaml")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--output", default="runs/tofu-openunlearning-reconstruction")
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-forget-sha256", required=True)
    parser.add_argument("--expected-original-model-sha256", required=True)
    parser.add_argument("--expected-original-optimizer-sha256", required=True)
    parser.add_argument("--expected-deletion-model-sha256", required=True)
    parser.add_argument("--expected-deletion-optimizer-sha256", required=True)
    args = parser.parse_args()

    verify_lock(args.sources, args.lock)
    config = copy.deepcopy(yaml.safe_load(Path(args.config).read_text()))
    if len(config.get("forget_scenarios", [])) != 1:
        raise RuntimeError("reconstruction requires exactly one forget scenario")
    if config.get("replay_policies") != ["slot_mask"]:
        raise RuntimeError("reconstruction requires slot_mask as the only replay policy")
    if bool(config.get("run_repacked_baseline", True)):
        raise RuntimeError("reconstruction refuses repacked baselines")

    model_cfg = config["model"]
    revision = revision_from_lock(args.lock, model_cfg["artifact_key"])
    model_cfg["revision"] = revision
    config["release_mode"] = True
    config["artifact_lock"] = args.lock

    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    device = torch.device(model_cfg.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical reconstruction requires CUDA")

    output = Path(args.output)
    runtime = output / "runtime"
    hf_root = output / "hf"
    runtime.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.resolved.json", config)
    save_json(output / "environment.json", environment_snapshot())

    model, tokenizer, optimizer = _new_model_optimizer(config)
    store = TokenStore(config["data"]["train_dir"], dummy_token_id=tokenizer.pad_token_id)
    plan_cfg = config["plan"]
    opt_cfg = config["optimizer"]
    plan = build_plan(
        store.ids,
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
    plan_path = runtime / "execution_plan.jsonl"
    plan_hash = write_plan(plan_path, plan)
    if plan_hash != args.expected_plan_sha256:
        raise RuntimeError(f"execution plan hash {plan_hash} != frozen {args.expected_plan_sha256}")

    base_checkpoint = runtime / "step-000000.pt"
    base_info = save_checkpoint(
        base_checkpoint,
        model,
        optimizer,
        next_optimizer_step=0,
        metadata={"config_sha256": config_sha256(config), "plan_sha256": plan_hash},
    )
    wal_path = runtime / "trace.wal"
    manifest_path = runtime / "manifest.jsonl"
    writer = WalWriter.from_environment(wal_path, manifest_path)
    runner = TraceRunner(model, optimizer, store, device=device, dtype=model_cfg.get("dtype", "fp32"))
    original_stats = runner.run(plan, policy="none", wal_writer=writer, progress_label="reconstruct-original")
    wal_summary = writer.close()
    original_model_hash = model_sha256(model)
    original_optimizer_hash = optimizer_sha256(optimizer)
    if original_model_hash != args.expected_original_model_sha256:
        raise RuntimeError(
            f"original model hash {original_model_hash} != frozen {args.expected_original_model_sha256}"
        )
    if original_optimizer_hash != args.expected_original_optimizer_sha256:
        raise RuntimeError(
            f"original optimizer hash {original_optimizer_hash} != frozen {args.expected_original_optimizer_sha256}"
        )
    export_hf(
        model,
        tokenizer,
        hf_root / "original",
        role="frozen_full_target",
        model_hash=original_model_hash,
        optimizer_hash=original_optimizer_hash,
        base_revision=revision,
    )
    original_payload = {
        "model_sha256": original_model_hash,
        "optimizer_sha256": original_optimizer_hash,
        "stats": original_stats.to_dict(),
        "base_checkpoint": base_info,
        "wal": wal_summary,
        "hf_checkpoint": str((hf_root / "original").resolve()),
    }
    save_json(output / "original.json", original_payload)
    release_phase(runner, optimizer, model, tokenizer)

    reconstructed = WalReader.from_environment(wal_path, manifest_path).to_plan()
    if reconstructed != read_plan(plan_path):
        raise RuntimeError("WAL reconstruction does not match the frozen execution plan")

    scenario = config["forget_scenarios"][0]
    forget_ids = select_scenario_forget_ids(
        plan,
        scenario,
        train_dir=config["data"]["train_dir"],
        seed=int(scenario.get("seed", seed + 100)),
    )
    forget_path = runtime / "forget_ids.txt"
    forget_path.write_text("\n".join(sorted(forget_ids)) + "\n")
    forget_hash = sha256_file(forget_path)
    if forget_hash != args.expected_forget_sha256:
        raise RuntimeError(f"forget-set hash {forget_hash} != frozen {args.expected_forget_sha256}")
    if len(forget_ids) != 400:
        raise RuntimeError(f"expected 400 forget10 records, observed {len(forget_ids)}")

    redaction = materialize_redacted_store(store, forget_ids, runtime / "redacted-data")
    if redaction.get("forgotten_ids_present") is not False:
        raise RuntimeError("physical redaction failed: forgotten IDs remain in replay store")
    if int(redaction.get("retained_records", -1)) != 3600:
        raise RuntimeError(f"expected 3600 retained records after forget10, got {redaction}")
    redacted_store = TokenStore(runtime / "redacted-data", dummy_token_id=store.dummy_token_id)

    deletion_model, deletion_tokenizer, deletion_optimizer = _new_model_optimizer(config)
    load_checkpoint(base_checkpoint, deletion_model, deletion_optimizer, map_location=device)
    deletion_runner = TraceRunner(
        deletion_model,
        deletion_optimizer,
        redacted_store,
        device=device,
        dtype=model_cfg.get("dtype", "fp32"),
    )
    deletion_stats = deletion_runner.run(
        reconstructed,
        forget_ids=forget_ids,
        policy="slot_mask",
        start_optimizer_step=0,
        progress_label="reconstruct-redacted-slot",
    )
    deletion_model_hash = model_sha256(deletion_model)
    deletion_optimizer_hash = optimizer_sha256(deletion_optimizer)
    if deletion_model_hash != args.expected_deletion_model_sha256:
        raise RuntimeError(
            f"deletion model hash {deletion_model_hash} != frozen {args.expected_deletion_model_sha256}"
        )
    if deletion_optimizer_hash != args.expected_deletion_optimizer_sha256:
        raise RuntimeError(
            f"deletion optimizer hash {deletion_optimizer_hash} != frozen {args.expected_deletion_optimizer_sha256}"
        )
    export_hf(
        deletion_model,
        deletion_tokenizer,
        hf_root / "deletion",
        role="frozen_physically_redacted_slot_replay",
        model_hash=deletion_model_hash,
        optimizer_hash=deletion_optimizer_hash,
        base_revision=revision,
    )
    deletion_payload = {
        "model_sha256": deletion_model_hash,
        "optimizer_sha256": deletion_optimizer_hash,
        "stats": deletion_stats.to_dict(),
        "forget_count": len(forget_ids),
        "forget_ids_sha256": forget_hash,
        "redaction": redaction,
        "hf_checkpoint": str((hf_root / "deletion").resolve()),
    }
    save_json(output / "deletion.json", deletion_payload)
    release_phase(deletion_runner, deletion_optimizer, deletion_model, deletion_tokenizer)

    final = {
        "status": "passed",
        "two_pass_reconstruction": True,
        "frozen_source_run": 31490644488,
        "frozen_source_commit": "c2375c1e491224062c05de6a3abe1b50f4af3937",
        "model_revision": revision,
        "plan_sha256": plan_hash,
        "forget_ids_sha256": forget_hash,
        "original": original_payload,
        "deletion": deletion_payload,
    }
    save_json(output / "summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
