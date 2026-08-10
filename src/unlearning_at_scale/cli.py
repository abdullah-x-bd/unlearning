from __future__ import annotations

import argparse
import json

from .config import load_config
from .experiment import run_experiment
from .smoke import run_core_smoke


def main() -> None:
    parser = argparse.ArgumentParser(prog="uas")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("core-smoke", help="run the dependency-light exact replay smoke test")
    smoke.add_argument("--output", default="runs/core-smoke")

    run = sub.add_parser("run", help="run one full LLM experiment config")
    run.add_argument("config")

    args = parser.parse_args()
    if args.command == "core-smoke":
        result = run_core_smoke(args.output)
    else:
        result = run_experiment(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
