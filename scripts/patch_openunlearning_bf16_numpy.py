from __future__ import annotations

import hashlib
import json
from pathlib import Path

UPSTREAM_COMMIT = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
TARGET = Path("external/open-unlearning/src/evals/metrics/utils.py")
MANIFEST = Path("results/openunlearning/bf16-numpy-patch.json")

REPLACEMENTS = {
    "avg_losses = avg_losses.cpu().numpy().tolist()":
        "avg_losses = avg_losses.float().cpu().numpy().tolist()",
    "normalized_probs = normalized_probs.cpu().numpy().tolist()":
        "normalized_probs = normalized_probs.float().cpu().numpy().tolist()",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"missing pinned OpenUnlearning target: {TARGET}")

    original = TARGET.read_bytes()
    text = original.decode("utf-8")
    for old, new in REPLACEMENTS.items():
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one pinned occurrence of {old!r}; found {count}"
            )
        text = text.replace(old, new)

    patched = text.encode("utf-8")
    TARGET.write_bytes(patched)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "applied",
        "purpose": "Represent BF16 scalar metric tensors as float32 before NumPy conversion; values are unchanged because BF16 values are exactly representable in float32.",
        "upstream_commit": UPSTREAM_COMMIT,
        "target": str(TARGET),
        "original_sha256": sha256_bytes(original),
        "patched_sha256": sha256_bytes(patched),
        "replacements": REPLACEMENTS,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
