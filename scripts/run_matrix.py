from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from unlearning_at_scale.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    args = parser.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    base = matrix["base"]
    output_root = Path(matrix["output_root"])
    summaries = []
    for model_entry in matrix["models"]:
        config = copy.deepcopy(base)
        config["model"].update(model_entry)
        slug = model_entry["name"].split("/")[-1]
        config["output_dir"] = str(output_root / slug)
        summaries.append({"model": model_entry["name"], "summary": run_experiment(config)})
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
