from __future__ import annotations

import argparse

from unlearning_at_scale.artifacts import freeze_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve remote revisions and hash prepared datasets")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--output", default="locks/artifacts.lock.json")
    args = parser.parse_args()
    freeze_artifacts(args.sources, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
