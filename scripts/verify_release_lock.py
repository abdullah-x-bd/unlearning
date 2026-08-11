from __future__ import annotations

import argparse

from unlearning_at_scale.artifacts import verify_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen publication artifact lock")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--skip-dataset-files", action="store_true")
    parser.add_argument("--skip-upstreams", action="store_true")
    args = parser.parse_args()
    verify_lock(args.sources, args.lock, verify_prepared_datasets=not args.skip_dataset_files, verify_upstreams=not args.skip_upstreams)
    print("release lock verified")


if __name__ == "__main__":
    main()
