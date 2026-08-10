from __future__ import annotations

import argparse
from pathlib import Path

from unlearning_at_scale.forget import content_closure, load_records_meta, near_duplicate_closure


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a deletion request over exact and near duplicates")
    parser.add_argument("train_dir")
    parser.add_argument("seed_ids")
    parser.add_argument("output")
    parser.add_argument("--exact-content", action="store_true")
    parser.add_argument("--near-hamming", type=int)
    args = parser.parse_args()
    metadata = load_records_meta(args.train_dir)
    ids = {line.strip() for line in Path(args.seed_ids).read_text().splitlines() if line.strip()}
    if args.exact_content:
        ids = content_closure(metadata, ids)
    if args.near_hamming is not None:
        ids = near_duplicate_closure(metadata, ids, args.near_hamming)
    Path(args.output).write_text("\n".join(sorted(ids)) + "\n")


if __name__ == "__main__":
    main()
