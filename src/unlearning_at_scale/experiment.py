from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from .compare import compare_state_dicts
from .config import config_sha256
from .dataset import TokenStore
from .determinism import configure_determinism, environment_snapshot
from .forget import earliest_forget_step, select_forget_ids
from .modeling import load_causal_lm
from .plan import build_plan, read_plan, write_plan
from .repacked import build_repacked_plan
from .state import load_checkpoint, model_sha256, optimizer_sha256, save_checkpoint
from .training import TraceRunner, create_optimizer
from .wal import WalReader, WalWriter


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _device(config: dict) -> torch.device:
    requested = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by the experiment config but is not available")
    return torch.device(requested)


def _optimizer_config(config: dict) -> tuple[float, float, bool, bool]:
    opt = config["optimizer"]
    return (
        float(opt["lr"]),
        float(opt.get("weight_decay", 0.0)),
        bool(opt.get("foreach", False)),
        bool(opt.get("fused", False)),
    )


def _new_model_optimizer(config: dict):
    model_cfg = config["model"]
    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    device = _device(config)
    model.to(device)
    lr, weight_decay, foreach, fused = _optimizer_config(config)
    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay, foreach=foreach, fused=fused)
    return model, tokenizer, optimizer


def _release(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_candidates(checkpoint_dir: Path, earliest_step: int) -> list[tuple[int, Path]]:
    candidates: dict[int, Path] = {0: checkpoint_dir / "step-000000.pt"}
    for path in checkpoint_dir.glob("step-*.pt"):
        try:
            step = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        if step <= earliest_step:
            candidates[step] = path
    return sorted(candidates.items())


def _comparison_to_saved_state(reference_path: Path, reference_hash: str, model: torch.nn.Module) -> dict:
    candidate_hash = model_sha256(model)
    if candidate_hash == reference_hash:
        state = model.state_dict()
        return {
            "exact": True,
            "total_tensors": len(state),
            "unequal_tensors": 0,
            "unequal_elements": 0,
            "total_elements": sum(t.numel() for t in state.values()),
            "max_abs_diff": 0.0,
            "l2_diff": 0.0,
            "left_sha256": reference_hash,
            "right_sha256": candidate_hash,
        }
    reference_state = torch.load(reference_path, map_location="cpu", weights_only=True)
    comparison = compare_state_dicts(reference_state, model.state_dict()).to_dict()
    del reference_state
    return comparison


def run_experiment(config: dict) -> dict:
    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    device = _device(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "config.resolved.json", config)
    _save_json(output_dir / "environment.json", environment_snapshot())

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
    plan_path = output_dir / "execution_plan.jsonl"
    plan_hash = write_plan(plan_path, plan)

    checkpoint_dir = output_dir / "original" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = checkpoint_dir / "step-000000.pt"
    base_info = save_checkpoint(
        base_checkpoint,
        model,
        optimizer,
        next_optimizer_step=0,
        metadata={"config_sha256": config_sha256(config), "plan_sha256": plan_hash},
    )

    wal_path = output_dir / "original" / "trace.wal"
    manifest_path = output_dir / "original" / "manifest.jsonl"
    writer = WalWriter.from_environment(wal_path, manifest_path)
    runner = TraceRunner(model, optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32"))
    original_stats = runner.run(
        plan,
        policy="none",
        wal_writer=writer,
        checkpoint_every=int(config.get("checkpoint_every", 0)) or None,
        checkpoint_dir=checkpoint_dir,
    )
    wal_summary = writer.close()
    original_model_hash = model_sha256(model)
    original_optimizer_hash = optimizer_sha256(optimizer)
    original_state_path = output_dir / "original" / "final-model-state.pt"
    torch.save(model.state_dict(), original_state_path)
    original_summary = {
        "stats": original_stats.to_dict(),
        "model_sha256": original_model_hash,
        "optimizer_sha256": original_optimizer_hash,
        "base_checkpoint": base_info,
        "final_model_state_bytes": original_state_path.stat().st_size,
        "wal": wal_summary,
    }
    _save_json(output_dir / "original" / "summary.json", original_summary)
    _release(runner, optimizer, model)

    reconstructed = WalReader.from_environment(wal_path, manifest_path).to_plan()
    if reconstructed != read_plan(plan_path):
        raise RuntimeError("WAL reconstruction does not match execution plan")

    identity_model, identity_tokenizer, identity_optimizer = _new_model_optimizer(config)
    load_checkpoint(base_checkpoint, identity_model, identity_optimizer, map_location=device)
    identity_runner = TraceRunner(identity_model, identity_optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32"))
    identity_stats = identity_runner.run(reconstructed, policy="none")
    identity_payload = {
        "stats": identity_stats.to_dict(),
        "model_comparison": _comparison_to_saved_state(original_state_path, original_model_hash, identity_model),
        "optimizer_hash_equal": optimizer_sha256(identity_optimizer) == original_optimizer_hash,
        "model_sha256": model_sha256(identity_model),
        "optimizer_sha256": optimizer_sha256(identity_optimizer),
    }
    _save_json(output_dir / "replay-identity" / "summary.json", identity_payload)
    _release(identity_runner, identity_optimizer, identity_model, identity_tokenizer)

    scenarios_out: list[dict] = []
    for scenario_index, scenario in enumerate(config.get("forget_scenarios", [])):
        scenario_name = scenario["name"]
        scenario_dir = output_dir / "forget" / scenario_name
        forget_ids = select_forget_ids(
            plan,
            fraction=float(scenario["fraction"]),
            strategy=scenario["strategy"],
            seed=int(scenario.get("seed", seed + 100 + scenario_index)),
        )
        (scenario_dir / "forget_ids.txt").parent.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "forget_ids.txt").write_text("\n".join(sorted(forget_ids)) + "\n")
        earliest = earliest_forget_step(plan, forget_ids)
        checkpoint_step, checkpoint_path = _checkpoint_candidates(checkpoint_dir, earliest)[-1]

        oracle_model, oracle_tokenizer, oracle_optimizer = _new_model_optimizer(config)
        load_checkpoint(checkpoint_path, oracle_model, oracle_optimizer, map_location=device)
        oracle_runner = TraceRunner(oracle_model, oracle_optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32"))
        oracle_stats = oracle_runner.run(plan, forget_ids=forget_ids, policy="slot_mask", start_optimizer_step=checkpoint_step)
        oracle_model_hash = model_sha256(oracle_model)
        oracle_optimizer_hash = optimizer_sha256(oracle_optimizer)
        oracle_state_path = scenario_dir / "oracle" / "final-model-state.pt"
        oracle_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(oracle_model.state_dict(), oracle_state_path)
        _save_json(
            scenario_dir / "oracle" / "summary.json",
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": str(checkpoint_path),
                "stats": oracle_stats.to_dict(),
                "model_sha256": oracle_model_hash,
                "optimizer_sha256": oracle_optimizer_hash,
                "final_model_state_bytes": oracle_state_path.stat().st_size,
            },
        )
        _release(oracle_runner, oracle_optimizer, oracle_model, oracle_tokenizer)

        policy_results: dict[str, dict] = {}
        for policy in config.get("replay_policies", ["slot_mask", "filter"]):
            replay_model, replay_tokenizer, replay_optimizer = _new_model_optimizer(config)
            load_checkpoint(checkpoint_path, replay_model, replay_optimizer, map_location=device)
            replay_runner = TraceRunner(replay_model, replay_optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32"))
            replay_stats = replay_runner.run(
                reconstructed,
                forget_ids=forget_ids,
                policy=policy,
                start_optimizer_step=checkpoint_step,
            )
            policy_payload = {
                "stats": replay_stats.to_dict(),
                "comparison_to_oracle": _comparison_to_saved_state(oracle_state_path, oracle_model_hash, replay_model),
                "model_sha256": model_sha256(replay_model),
                "optimizer_sha256": optimizer_sha256(replay_optimizer),
                "optimizer_hash_equal_to_oracle": optimizer_sha256(replay_optimizer) == oracle_optimizer_hash,
            }
            _save_json(scenario_dir / f"replay-{policy}" / "summary.json", policy_payload)
            policy_results[policy] = policy_payload
            _release(replay_runner, replay_optimizer, replay_model, replay_tokenizer)

        repacked_payload = None
        if bool(config.get("run_repacked_baseline", True)):
            repacked_model, repacked_tokenizer, repacked_optimizer = _new_model_optimizer(config)
            load_checkpoint(checkpoint_path, repacked_model, repacked_optimizer, map_location=device)
            repacked_plan = build_repacked_plan(
                plan,
                forget_ids,
                start_optimizer_step=checkpoint_step,
                microbatch_size=int(plan_cfg["microbatch_size"]),
                grad_accum_steps=int(plan_cfg["grad_accum_steps"]),
                rng_seed=int(plan_cfg.get("rng_seed", seed + 1)) + 5000,
                peak_lr=float(opt_cfg["lr"]),
                warmup_ratio=float(opt_cfg.get("warmup_ratio", 0.0)),
                schedule=opt_cfg.get("schedule", "constant"),
            )
            repacked_runner = TraceRunner(repacked_model, repacked_optimizer, store, device=device, dtype=config["model"].get("dtype", "fp32"))
            repacked_stats = repacked_runner.run(repacked_plan, policy="none")
            repacked_payload = {
                "stats": repacked_stats.to_dict(),
                "comparison_to_trace_oracle": _comparison_to_saved_state(oracle_state_path, oracle_model_hash, repacked_model),
                "model_sha256": model_sha256(repacked_model),
                "optimizer_sha256": optimizer_sha256(repacked_optimizer),
                "optimizer_hash_equal_to_oracle": optimizer_sha256(repacked_optimizer) == oracle_optimizer_hash,
            }
            _save_json(scenario_dir / "repacked" / "summary.json", repacked_payload)
            _release(repacked_runner, repacked_optimizer, repacked_model, repacked_tokenizer)

        scenario_payload = {
            "name": scenario_name,
            "forget_count": len(forget_ids),
            "earliest_affected_step": earliest,
            "checkpoint_step": checkpoint_step,
            "replay": policy_results,
            "repacked": repacked_payload,
        }
        _save_json(scenario_dir / "summary.json", scenario_payload)
        scenarios_out.append(scenario_payload)

    summary = {
        "config_sha256": config_sha256(config),
        "plan_sha256": plan_hash,
        "original": original_summary,
        "replay_identity": identity_payload,
        "scenarios": scenarios_out,
    }
    _save_json(output_dir / "summary.json", summary)
    return summary
