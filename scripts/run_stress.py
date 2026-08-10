from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from unlearning_at_scale.config import load_config
from unlearning_at_scale.experiment import run_experiment


def set_nested(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stress_config")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.stress_config).read_text())
    base = load_config(spec["base_config"])
    output_root = Path(spec["output_root"])
    summaries = []
    for variant in spec["variants"]:
        config = copy.deepcopy(base)
        for key, value in variant.get("overrides", {}).items():
            set_nested(config, key, value)
        config["output_dir"] = str(output_root / variant["name"])
        summaries.append({"name": variant["name"], "summary": run_experiment(config)})
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stress_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
