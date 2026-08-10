from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning_at_scale.dataset import TokenStore, materialize_redacted_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Physically omit a forget set from a prepared token store")
    parser.add_argument("source")
    parser.add_argument("forget_ids")
    parser.add_argument("output")
    args = parser.parse_args()
    forget = {line.strip() for line in Path(args.forget_ids).read_text().splitlines() if line.strip()}
    payload = materialize_redacted_store(TokenStore(args.source), forget, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
