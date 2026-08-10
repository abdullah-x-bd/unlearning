from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning_at_scale.benchmark import benchmark_checkpoint_patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoints = sorted(Path(args.checkpoint_dir).glob("step-*.pt"))
    if len(checkpoints) < 2:
        raise SystemExit("Need at least two checkpoints")
    rows = []
    for before, after in zip(checkpoints, checkpoints[1:]):
        row = benchmark_checkpoint_patch(before, after)
        row["before"] = before.name
        row["after"] = after.name
        rows.append(row)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
