from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scaling matrix from a verified artifact lock")
    parser.add_argument("matrix")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    args = parser.parse_args()
    verify_lock(args.sources, args.lock)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    base = matrix["base"]
    output_root = Path(matrix["output_root"])
    summaries = []
    for model_entry in matrix["models"]:
        config = copy.deepcopy(base)
        config["model"].update(model_entry)
        artifact_key = config["model"].get("artifact_key")
        if not artifact_key:
            raise ValueError(f"missing artifact_key for {model_entry['name']}")
        config["model"]["revision"] = revision_from_lock(args.lock, artifact_key)
        config["release_mode"] = True
        config["artifact_lock"] = args.lock
        slug = model_entry["name"].split("/")[-1]
        config["output_dir"] = str(output_root / slug)
        summaries.append({"model": model_entry["name"], "summary": run_experiment(config)})
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
