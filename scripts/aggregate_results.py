from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning_at_scale.results import collect_run_rows, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--csv", default="results/experiment_rows.csv")
    parser.add_argument("--json", default="results/experiment_rows.json")
    args = parser.parse_args()
    rows = collect_run_rows(args.root)
    write_rows_csv(rows, args.csv)
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"aggregated {len(rows)} summaries")


if __name__ == "__main__":
    main()
