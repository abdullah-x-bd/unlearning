from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml

TOFU_SPLITS = {
    "forget01": ("holdout01", "retain99"),
    "forget05": ("holdout05", "retain95"),
    "forget10": ("holdout10", "retain90"),
}

RETAIN_LOCK_KEYS = {
    "retain99": "tofu_llama32_1b_retain99_reference",
    "retain95": "tofu_llama32_1b_retain95_reference",
    "retain90": "tofu_llama32_1b_retain90_reference",
}


def load_upstream(lock_path: Path, name: str = "open_unlearning") -> tuple[Path, str]:
    payload = yaml.safe_load(lock_path.read_text())
    spec = payload["upstreams"][name]
    checkout = Path(spec["checkout"])
    expected = str(spec["commit"])
    observed = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != expected:
        raise RuntimeError(f"{name} is {observed}, expected {expected}")
    return checkout, expected


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tofu_model_overrides(checkpoint: str | Path, attention_implementation: str) -> list[str]:
    checkpoint_text = str(checkpoint)
    return [
        f"model.model_args.pretrained_model_name_or_path={checkpoint_text}",
        f"model.tokenizer_args.pretrained_model_name_or_path={checkpoint_text}",
        f"model.model_args.attn_implementation={attention_implementation}",
    ]


def pinned_retain_reference(
    args: argparse.Namespace, retain_split: str, retain_dir: Path
) -> tuple[str, dict]:
    if args.retain_checkpoint:
        path = str(Path(args.retain_checkpoint).resolve())
        return path, {"kind": "explicit_local_checkpoint", "path": path}

    lock = json.loads(Path(args.artifact_lock).read_text())
    key = RETAIN_LOCK_KEYS[retain_split]
    item = lock["huggingface"][key]
    provenance = {
        "kind": "pinned_huggingface_snapshot",
        "artifact_key": key,
        "repo_id": item["repo_id"],
        "revision": item["resolved_sha"],
    }
    local_dir = (retain_dir / "checkpoint").resolve()
    if args.dry_run:
        provenance["planned_local_dir"] = str(local_dir)
        return str(local_dir), provenance

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=item["repo_id"],
        revision=item["resolved_sha"],
        local_dir=local_dir,
    )
    provenance["local_dir"] = str(local_dir)
    return str(local_dir), provenance


def tofu_eval(args: argparse.Namespace) -> None:
    checkout, commit = load_upstream(Path(args.upstreams))
    holdout_split, retain_split = TOFU_SPLITS[args.forget_split]
    retain_dir = Path(args.output_root) / "reference" / retain_split
    retain_model, retain_provenance = pinned_retain_reference(
        args, retain_split, retain_dir
    )
    retain_log = retain_dir / "TOFU_EVAL.json"
    commands: list[list[str]] = []

    if not retain_log.exists():
        command = [
            "python",
            "src/eval.py",
            "--config-name=eval.yaml",
            "experiment=eval/tofu/default",
            f"forget_split={args.forget_split}",
            f"holdout_split={holdout_split}",
            f"model={args.model}",
            f"task_name=uas_reference_{args.model}_{retain_split}",
            *tofu_model_overrides(retain_model, args.attention_implementation),
            f"paths.output_dir={retain_dir.resolve()}",
        ]
        commands.append(command)
        run(command, checkout, args.dry_run)

    task = args.task_name or f"uas_replay_{args.model}_{args.forget_split}"
    result_dir = Path(args.output_root) / "models" / task
    checkpoint = Path(args.checkpoint).resolve()
    command = [
        "python",
        "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        f"forget_split={args.forget_split}",
        f"holdout_split={holdout_split}",
        f"model={args.model}",
        f"task_name={task}",
        *tofu_model_overrides(checkpoint, args.attention_implementation),
        f"paths.output_dir={result_dir.resolve()}",
        f"retain_logs_path={retain_log.resolve()}",
    ]
    commands.append(command)
    run(command, checkout, args.dry_run)
    save_manifest(
        result_dir / "uas_interop.json",
        {
            "framework": "OpenUnlearning",
            "framework_commit": commit,
            "benchmark": "TOFU",
            "model": args.model,
            "checkpoint": str(checkpoint),
            "forget_split": args.forget_split,
            "holdout_split": holdout_split,
            "retain_split": retain_split,
            "retain_reference": retain_provenance,
            "attention_implementation": args.attention_implementation,
            "tokenizer_source": "same checkpoint as each evaluated model",
            "commands": commands,
        },
    )


def tofu_baselines(args: argparse.Namespace) -> None:
    checkout, commit = load_upstream(Path(args.upstreams))
    holdout_split, retain_split = TOFU_SPLITS[args.forget_split]
    methods = args.methods or ["GradAscent", "GradDiff", "NPO", "SimNPO"]
    model_path = (
        str(Path(args.target_checkpoint).resolve())
        if args.target_checkpoint
        else f"open-unlearning/tofu_{args.model}_full"
    )
    retain_logs = (
        args.retain_logs
        or f"saves/eval/tofu_{args.model}_{retain_split}/TOFU_EVAL.json"
    )
    manifest = {
        "framework": "OpenUnlearning",
        "framework_commit": commit,
        "benchmark": "TOFU",
        "model": args.model,
        "target": model_path,
        "forget_split": args.forget_split,
        "retain_split": retain_split,
        "methods": methods,
        "commands": [],
    }

    for method in methods:
        task = f"uas_tofu_{args.model}_{args.forget_split}_{method}"
        command = [
            "python",
            "src/train.py",
            "--config-name=unlearn.yaml",
            "experiment=unlearn/tofu/default",
            f"trainer={method}",
            f"task_name={task}",
            f"model={args.model}",
            f"forget_split={args.forget_split}",
            f"retain_split={retain_split}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"retain_logs_path={retain_logs}",
        ]
        run(command, checkout, args.dry_run)
        manifest["commands"].append(command)

        eval_command = [
            "python",
            "src/eval.py",
            "--config-name=eval.yaml",
            "experiment=eval/tofu/default",
            f"forget_split={args.forget_split}",
            f"holdout_split={holdout_split}",
            f"model={args.model}",
            f"task_name={task}",
            f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{task}",
            f"paths.output_dir=saves/unlearn/{task}/evals",
            f"retain_logs_path={retain_logs}",
        ]
        run(eval_command, checkout, args.dry_run)
        manifest["commands"].append(eval_command)

    save_manifest(
        Path(args.output_root) / "official_baselines_manifest.json", manifest
    )


def muse(args: argparse.Namespace) -> None:
    checkout, commit = load_upstream(Path(args.upstreams))
    methods = args.methods or ["GradAscent", "GradDiff", "NPO", "SimNPO"]
    retain_logs = (
        f"saves/eval/muse_Llama-2-7b-hf_{args.data_split}_retrain/MUSE_EVAL.json"
    )
    payload = {
        "framework": "OpenUnlearning",
        "framework_commit": commit,
        "benchmark": "MUSE",
        "model": "Llama-2-7b-hf",
        "data_split": args.data_split,
        "methods": methods,
        "budget_dependent": True,
        "commands": [],
    }

    for method in methods:
        task = f"uas_muse_Llama-2-7b-hf_{args.data_split}_{method}"
        command = [
            "python",
            "src/train.py",
            "--config-name=unlearn.yaml",
            "experiment=unlearn/muse/default",
            "model=Llama-2-7b-hf",
            f"data_split={args.data_split}",
            f"trainer={method}",
            f"task_name={task}",
            f"retain_logs_path={retain_logs}",
        ]
        payload["commands"].append(command)
        run(command, checkout, args.dry_run)

        eval_command = [
            "python",
            "src/eval.py",
            "--config-name=eval.yaml",
            "experiment=eval/muse/default",
            f"data_split={args.data_split}",
            f"task_name={task}",
            "model=Llama-2-7b-hf",
            f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{task}",
            f"paths.output_dir=saves/unlearn/{task}/evals",
            f"retain_logs_path={retain_logs}",
        ]
        payload["commands"].append(eval_command)
        run(eval_command, checkout, args.dry_run)

    save_manifest(Path(args.output_root) / "muse_manifest.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pinned OpenUnlearning interoperability adapter"
    )
    parser.add_argument("--upstreams", default="external/upstreams.lock.yaml")
    parser.add_argument("--artifact-lock", default="locks/artifacts.lock.json")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("tofu-eval")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument(
        "--forget-split", choices=sorted(TOFU_SPLITS), required=True
    )
    ev.add_argument("--model", default="Llama-3.2-1B-Instruct")
    ev.add_argument("--retain-checkpoint")
    ev.add_argument("--task-name")
    ev.add_argument("--output-root", default="results/openunlearning/tofu")
    ev.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="Explicit attention backend used uniformly for the retain reference and evaluated checkpoints.",
    )
    ev.add_argument("--dry-run", action="store_true")
    ev.set_defaults(func=tofu_eval)

    bl = sub.add_parser("tofu-baselines")
    bl.add_argument(
        "--forget-split", choices=sorted(TOFU_SPLITS), required=True
    )
    bl.add_argument("--model", default="Llama-3.2-1B-Instruct")
    bl.add_argument("--methods", nargs="+")
    bl.add_argument("--target-checkpoint")
    bl.add_argument("--retain-logs")
    bl.add_argument("--output-root", default="results/openunlearning/tofu")
    bl.add_argument("--dry-run", action="store_true")
    bl.set_defaults(func=tofu_baselines)

    mu = sub.add_parser("muse")
    mu.add_argument("--data-split", choices=["News", "Books"], required=True)
    mu.add_argument("--methods", nargs="+")
    mu.add_argument("--output-root", default="results/openunlearning/muse")
    mu.add_argument("--dry-run", action="store_true")
    mu.set_defaults(func=muse)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
