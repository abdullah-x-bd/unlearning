from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from unlearning_at_scale.duplicates import simhash64


def row_key(question: str, answer: str) -> str:
    return hashlib.sha256((question + "\0" + answer).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TOFU as an immutable token store plus official forget/retain ID lists")
    parser.add_argument("--model", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install LLM dependencies first: pip install -e '.[llm]'") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    full = load_dataset("locuslab/TOFU", "full", split="train", revision=args.revision)
    split_names = ["forget01", "retain99", "forget05", "retain95", "forget10", "retain90"]
    split_keys: dict[str, set[str]] = {}
    for split_name in split_names:
        ds = load_dataset("locuslab/TOFU", split_name, split="train", revision=args.revision)
        split_keys[split_name] = {row_key(str(row["question"]), str(row["answer"])) for row in ds}

    ids: list[str] = []
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    metadata: list[dict] = []
    key_to_id: dict[str, str] = {}
    for index, row in enumerate(full):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        key = row_key(question, answer)
        sample_id = f"tofu-{index:04d}-{key[:12]}"
        text = f"Question: {question}\nAnswer: {answer}"
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.sequence_length,
            padding="max_length",
            return_attention_mask=True,
        )
        ids.append(sample_id)
        input_ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
        key_to_id[key] = sample_id
        metadata.append(
            {
                "sample_id": sample_id,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "simhash64": simhash64(text),
                "question": question,
                "answer": answer,
                "tofu_key": key,
            }
        )

    np.save(output / "input_ids.npy", np.asarray(input_ids, dtype=np.int64))
    np.save(output / "attention_mask.npy", np.asarray(masks, dtype=np.int64))
    (output / "ids.json").write_text(json.dumps(ids))
    (output / "records_meta.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata))

    split_counts = {}
    for split_name, keys in split_keys.items():
        selected = sorted(key_to_id[key] for key in keys if key in key_to_id)
        (output / f"{split_name}_ids.txt").write_text("\n".join(selected) + "\n")
        split_counts[split_name] = len(selected)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    files = ["input_ids.npy", "attention_mask.npy", "ids.json", "records_meta.jsonl"] + [f"{name}_ids.txt" for name in split_names]
    manifest = {
        "dataset": "locuslab/TOFU",
        "config": "full",
        "revision": args.revision,
        "tokenizer": args.model,
        "sequence_length": args.sequence_length,
        "records": len(ids),
        "split_counts": split_counts,
        "dataset_fingerprint": getattr(full, "_fingerprint", None),
        "files": {name: sha256_file(output / name) for name in files},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
