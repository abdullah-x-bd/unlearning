from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from unlearning_at_scale.duplicates import simhash64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable fixed-length token dataset for replay experiments")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--subset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True, help="Immutable dataset commit SHA")
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--model-revision", required=True, help="Immutable tokenizer/model commit SHA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--canary-groups", type=int, default=32)
    parser.add_argument("--canary-repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install LLM dependencies first: pip install -e '.[llm]'") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset, args.subset, split=args.split, revision=args.revision)
    texts: list[tuple[str, str, str]] = []
    for row_index, row in enumerate(dataset):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        texts.append((f"source-{row_index:08d}", text, content_hash))
        if len(texts) >= args.max_records:
            break

    rng = random.Random(args.seed)
    canaries: list[dict] = []
    for group in range(args.canary_groups):
        secret = "-".join(f"{rng.getrandbits(32):08x}" for _ in range(3))
        prompt = f"UAS synthetic canary {group:04d} secret is"
        completion = f" {secret}"
        text = prompt + completion
        digest = hashlib.sha256(text.encode()).hexdigest()
        group_ids: list[str] = []
        for rep in range(args.canary_repetitions):
            sample_id = f"canary-{group:04d}-rep-{rep:03d}"
            texts.append((sample_id, text, digest))
            group_ids.append(sample_id)
        canaries.append({"group": group, "prompt": prompt, "completion": completion, "sample_ids": group_ids, "content_sha256": digest})

    ids: list[str] = []
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    metadata: list[dict] = []
    for sample_id, text, content_hash in texts:
        encoded = tokenizer(text, truncation=True, max_length=args.sequence_length, padding="max_length", return_attention_mask=True)
        ids.append(sample_id)
        input_ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
        metadata.append({"sample_id": sample_id, "content_sha256": content_hash, "simhash64": simhash64(text), "is_canary": sample_id.startswith("canary-")})

    np.save(output / "input_ids.npy", np.asarray(input_ids, dtype=np.int64))
    np.save(output / "attention_mask.npy", np.asarray(masks, dtype=np.int64))
    (output / "ids.json").write_text(json.dumps(ids))
    (output / "records_meta.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata))
    (output / "canaries.json").write_text(json.dumps(canaries, indent=2, sort_keys=True) + "\n")

    files = ["input_ids.npy", "attention_mask.npy", "ids.json", "records_meta.jsonl", "canaries.json"]
    manifest = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "dataset_revision": args.revision,
        "tokenizer": args.model,
        "tokenizer_revision": args.model_revision,
        "sequence_length": args.sequence_length,
        "records": len(ids),
        "source_records": len(ids) - args.canary_groups * args.canary_repetitions,
        "canary_groups": args.canary_groups,
        "canary_repetitions": args.canary_repetitions,
        "seed": args.seed,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "files": {name: sha256_file(output / name) for name in files},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
