from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from unlearning_at_scale.determinism import environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = environment_snapshot()
    payload["pip_freeze"] = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True).splitlines()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
