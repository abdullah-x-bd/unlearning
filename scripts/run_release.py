from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from unlearning_at_scale import experiment
from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.lifecycle import release_phase


def resolve_config(config: dict, lock_path: str) -> dict:
    resolved = copy.deepcopy(config)
    model_cfg = resolved["model"]
    artifact_key = model_cfg.get("artifact_key")
    if not artifact_key:
        raise ValueError("release config requires model.artifact_key")
    model_cfg["revision"] = revision_from_lock(lock_path, artifact_key)
    resolved["release_mode"] = True
    resolved["artifact_lock"] = lock_path
    return resolved


def run_release_phase_smoke(config_path: str, config: dict, lock_path: str, sources_path: str) -> None:
    if not bool(config.get("release_phase_smoke", False)):
        return
    output_dir = Path(config["output_dir"]) / "preflight" / "gpu-phase-release-smoke.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/gpu_phase_release_smoke.py",
            "--config",
            config_path,
            "--lock",
            lock_path,
            "--sources",
            sources_path,
            "--output",
            str(output_dir),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an exact experiment only from a verified artifact lock")
    parser.add_argument("config")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    args = parser.parse_args()
    verify_lock(args.sources, args.lock)
    config = yaml.safe_load(Path(args.config).read_text())
    experiment._release = release_phase
    run_release_phase_smoke(args.config, config, args.lock, args.sources)
    result = experiment.run_experiment(resolve_config(config, args.lock))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
