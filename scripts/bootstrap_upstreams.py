from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_checkout(repository: str, commit: str, checkout: Path) -> None:
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        run(["git", "clone", "--filter=blob:none", repository, str(checkout)])
    run(["git", "-C", str(checkout), "fetch", "origin", commit])
    run(["git", "-C", str(checkout), "checkout", "--detach", commit])
    observed = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if observed != commit:
        raise RuntimeError(f"{checkout} resolved to {observed}, expected {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify pinned external research frameworks")
    parser.add_argument("--lock", default="external/upstreams.lock.yaml")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.lock).read_text())
    for name, spec in payload["upstreams"].items():
        print(f"Pinning {name} at {spec['commit']}")
        ensure_checkout(str(spec["repository"]), str(spec["commit"]), Path(spec["checkout"]))


if __name__ == "__main__":
    main()
