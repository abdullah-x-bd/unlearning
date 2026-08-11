from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful assistant."
DATE_STRING = "10 Apr 2025"
SPLITS = ["forget01", "retain99", "forget05", "retain95", "forget10", "retain90"]


def row_key(question: str, answer: str) -> str:
    return hashlib.sha256((question + "\0" + answer).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_example(tokenizer, question: str, answer: str, max_length: int):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    chat_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, date_string=DATE_STRING)
    prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, date_string=DATE_STRING)
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids = list(chat_ids) + [tokenizer.eos_token_id]
    if len(chat_ids) > max_length:
        raise ValueError(
            f"TOFU example has {len(chat_ids)} tokens, above max_length={max_length}; "
            "increase max_length rather than silently changing upstream preprocessing"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(chat_ids[len(prompt_ids):])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padding = max_length - len(chat_ids)
    return (
        list(chat_ids) + [pad_id] * padding,
        [1] * len(chat_ids) + [0] * padding,
        labels + [IGNORE_INDEX] * padding,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TOFU with pinned OpenUnlearning Llama-3.2 chat and label semantics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install LLM dependencies first: pip install -e '.[llm]'") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    full = load_dataset("locuslab/TOFU", "full", split="train", revision=args.dataset_revision)
    split_keys = {}
    for split_name in SPLITS:
        ds = load_dataset("locuslab/TOFU", split_name, split="train", revision=args.dataset_revision)
        split_keys[split_name] = {row_key(str(row["question"]), str(row["answer"])) for row in ds}

    ids = []
    input_rows = []
    attention_rows = []
    label_rows = []
    metadata = []
    key_to_id = {}
    for index, row in enumerate(full):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        key = row_key(question, answer)
        sample_id = f"tofu-{index:04d}-{key[:12]}"
        input_ids, attention_mask, labels = encode_example(tokenizer, question, answer, args.max_length)
        ids.append(sample_id)
        input_rows.append(input_ids)
        attention_rows.append(attention_mask)
        label_rows.append(labels)
        key_to_id[key] = sample_id
        metadata.append({
            "sample_id": sample_id,
            "content_sha256": hashlib.sha256((question + "\n" + answer).encode()).hexdigest(),
            "question": question,
            "answer": answer,
            "target_tokens": sum(value != IGNORE_INDEX for value in labels),
        })

    np.save(output / "input_ids.npy", np.asarray(input_rows, dtype=np.int64))
    np.save(output / "attention_mask.npy", np.asarray(attention_rows, dtype=np.int64))
    np.save(output / "labels.npy", np.asarray(label_rows, dtype=np.int64))
    (output / "ids.json").write_text(json.dumps(ids))
    (output / "records_meta.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata))

    split_counts = {}
    for split_name, keys in split_keys.items():
        selected = sorted(key_to_id[key] for key in keys if key in key_to_id)
        (output / f"{split_name}_ids.txt").write_text("\n".join(selected) + "\n")
        split_counts[split_name] = len(selected)

    files = ["input_ids.npy", "attention_mask.npy", "labels.npy", "ids.json", "records_meta.jsonl", *[f"{name}_ids.txt" for name in SPLITS]]
    manifest = {
        "dataset": "locuslab/TOFU",
        "dataset_revision": args.dataset_revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "openunlearning_commit": "4ad738aaf60f6a4385f6e2506d01da99e76c31f3",
        "template": {
            "apply_chat_template": True,
            "system_prompt": SYSTEM_PROMPT,
            "date_string": DATE_STRING,
            "loss_on": "final assistant response only",
            "ignore_index": IGNORE_INDEX,
        },
        "max_length": args.max_length,
        "records": len(ids),
        "split_counts": split_counts,
        "dataset_fingerprint": getattr(full, "_fingerprint", None),
        "files": {name: sha256_file(output / name) for name in files},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
