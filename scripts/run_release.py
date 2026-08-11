from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.experiment import run_experiment


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an exact experiment only from a verified artifact lock")
    parser.add_argument("config")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    args = parser.parse_args()
    verify_lock(args.sources, args.lock)
    config = yaml.safe_load(Path(args.config).read_text())
    result = run_experiment(resolve_config(config, args.lock))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
