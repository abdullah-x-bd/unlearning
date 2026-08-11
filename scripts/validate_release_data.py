from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        observed = sha256_file(directory / name)
        if observed != expected:
            raise RuntimeError(f"manifest hash mismatch for {directory / name}: {observed} != {expected}")
    return manifest


def validate_wikitext(directory: Path) -> dict:
    manifest = verify_manifest(directory)
    ids = json.loads((directory / "ids.json").read_text())
    input_ids = np.load(directory / "input_ids.npy", mmap_mode="r")
    masks = np.load(directory / "attention_mask.npy", mmap_mode="r")
    if len(ids) != len(set(ids)):
        raise RuntimeError("WikiText token store contains duplicate sample IDs")
    if input_ids.shape != masks.shape or input_ids.shape[0] != len(ids):
        raise RuntimeError("WikiText arrays and IDs disagree")
    if input_ids.shape[1] != int(manifest["sequence_length"]):
        raise RuntimeError("WikiText sequence length disagrees with manifest")
    expected_canaries = int(manifest["canary_groups"]) * int(manifest["canary_repetitions"])
    canary_ids = [sample_id for sample_id in ids if sample_id.startswith("canary-")]
    if len(canary_ids) != expected_canaries:
        raise RuntimeError(f"expected {expected_canaries} canary rows, observed {len(canary_ids)}")
    if int(manifest["records"]) != len(ids):
        raise RuntimeError("WikiText record count disagrees with manifest")
    return {
        "records": len(ids),
        "source_records": int(manifest["source_records"]),
        "canary_records": len(canary_ids),
        "sequence_length": int(input_ids.shape[1]),
    }


def validate_tofu(directory: Path) -> dict:
    manifest = verify_manifest(directory)
    ids = json.loads((directory / "ids.json").read_text())
    id_set = set(ids)
    if len(ids) != len(id_set):
        raise RuntimeError("TOFU token store contains duplicate sample IDs")
    input_ids = np.load(directory / "input_ids.npy", mmap_mode="r")
    masks = np.load(directory / "attention_mask.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    if input_ids.shape != masks.shape or input_ids.shape != labels.shape or input_ids.shape[0] != len(ids):
        raise RuntimeError("TOFU arrays and IDs disagree")
    if input_ids.shape[1] != int(manifest["max_length"]):
        raise RuntimeError("TOFU sequence length disagrees with manifest")
    active = labels != IGNORE_INDEX
    if np.any(active & (labels != input_ids)):
        raise RuntimeError("TOFU active labels do not equal the corresponding assistant-response tokens")
    active_per_row = active.sum(axis=1)
    if np.any(active_per_row <= 0):
        raise RuntimeError("TOFU contains a row with no supervised assistant-response token")
    if np.any(active & (masks == 0)):
        raise RuntimeError("TOFU contains supervised labels on padding positions")

    expected_counts = {
        "forget01": 40,
        "retain99": 3960,
        "forget05": 200,
        "retain95": 3800,
        "forget10": 400,
        "retain90": 3600,
    }
    split_sets: dict[str, set[str]] = {}
    for name, expected in expected_counts.items():
        values = read_ids(directory / f"{name}_ids.txt")
        split_sets[name] = set(values)
        if len(values) != expected or len(split_sets[name]) != expected:
            raise RuntimeError(f"TOFU {name} expected {expected} unique rows, observed {len(split_sets[name])}")
        if not split_sets[name].issubset(id_set):
            raise RuntimeError(f"TOFU {name} contains IDs absent from the full token store")
        if int(manifest["split_counts"][name]) != expected:
            raise RuntimeError(f"TOFU manifest reports the wrong count for {name}")

    for forget, retain in [("forget01", "retain99"), ("forget05", "retain95"), ("forget10", "retain90")]:
        if split_sets[forget] & split_sets[retain]:
            raise RuntimeError(f"TOFU {forget} and {retain} overlap")
        if split_sets[forget] | split_sets[retain] != id_set:
            raise RuntimeError(f"TOFU {forget} and {retain} do not partition the full set")

    if len(ids) != 4000 or int(manifest["records"]) != 4000:
        raise RuntimeError(f"TOFU full set expected 4000 rows, observed {len(ids)}")
    return {
        "records": len(ids),
        "shape": list(input_ids.shape),
        "target_tokens_min": int(active_per_row.min()),
        "target_tokens_max": int(active_per_row.max()),
        "target_tokens_mean": float(active_per_row.mean()),
        "split_counts": expected_counts,
    }


def main() -> None:
    report = {
        "wikitext": validate_wikitext(Path("data/prepared/wikitext103-pythia-256")),
        "tofu": validate_tofu(Path("data/prepared/tofu-llama32-1b-openunlearning")),
    }
    output = Path("results/cpu-preflight-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
