from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

IGNORE_INDEX = -100


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sample_weights: torch.Tensor
    sample_ids: tuple[str, ...]
    retained_count: int


class TokenStore:
    def __init__(self, directory: str | Path, dummy_token_id: int = 0):
        self.directory = Path(directory)
        self.ids = json.loads((self.directory / "ids.json").read_text())
        self.id_to_row = {sample_id: index for index, sample_id in enumerate(self.ids)}
        self.input_ids = np.load(self.directory / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(self.directory / "attention_mask.npy", mmap_mode="r")
        labels_path = self.directory / "labels.npy"
        self.labels = np.load(labels_path, mmap_mode="r") if labels_path.exists() else None
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes differ")
        if self.labels is not None and self.labels.shape != self.input_ids.shape:
            raise ValueError("labels and input_ids shapes differ")
        if self.input_ids.shape[0] != len(self.ids):
            raise ValueError("ids.json does not match token arrays")
        self.sequence_length = int(self.input_ids.shape[1])
        self.dummy_token_id = int(dummy_token_id)

    def __len__(self) -> int:
        return len(self.ids)

    def get_batch(self, sample_ids: Iterable[str], forget_ids: set[str] | None = None, policy: str = "none") -> TokenBatch:
        forget = forget_ids or set(); ordered = tuple(sample_ids)
        if policy not in {"none", "filter", "slot_mask"}:
            raise ValueError(f"unknown replay policy: {policy}")
        rows = []; masks = []; labels = []; weights = []; output_ids = []
        for sample_id in ordered:
            is_forgotten = sample_id in forget
            if is_forgotten and policy == "filter":
                continue
            if is_forgotten and policy == "slot_mask":
                dummy = np.full((self.sequence_length,), self.dummy_token_id, dtype=np.int64)
                rows.append(dummy); masks.append(np.ones((self.sequence_length,), dtype=np.int64)); labels.append(dummy.copy()); weights.append(0.0); output_ids.append(sample_id)
                continue
            row = self.id_to_row.get(sample_id)
            if row is None:
                raise KeyError(f"sample ID not found in token store: {sample_id}")
            input_row = np.asarray(self.input_ids[row], dtype=np.int64)
            rows.append(input_row); masks.append(np.asarray(self.attention_mask[row], dtype=np.int64))
            labels.append(input_row.copy() if self.labels is None else np.asarray(self.labels[row], dtype=np.int64))
            weights.append(1.0); output_ids.append(sample_id)
        if not rows:
            empty = torch.empty((0, self.sequence_length), dtype=torch.long)
            return TokenBatch(empty, empty.clone(), empty.clone(), torch.empty((0,), dtype=torch.float32), tuple(), 0)
        return TokenBatch(
            input_ids=torch.from_numpy(np.stack(rows, axis=0).copy()).long(),
            attention_mask=torch.from_numpy(np.stack(masks, axis=0).copy()).long(),
            labels=torch.from_numpy(np.stack(labels, axis=0).copy()).long(),
            sample_weights=torch.tensor(weights, dtype=torch.float32),
            sample_ids=tuple(output_ids),
            retained_count=sum(weight > 0 for weight in weights),
        )

    def redact(self, forget_ids: set[str]) -> "RedactedTokenStore":
        return RedactedTokenStore(self, forget_ids)


def materialize_redacted_store(source: TokenStore, forget_ids: set[str], output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    retained_ids = [sample_id for sample_id in source.ids if sample_id not in forget_ids]
    retained_rows = [source.id_to_row[sample_id] for sample_id in retained_ids]
    np.save(output / "input_ids.npy", np.asarray(source.input_ids[retained_rows], dtype=np.int64)); np.save(output / "attention_mask.npy", np.asarray(source.attention_mask[retained_rows], dtype=np.int64))
    if source.labels is not None:
        np.save(output / "labels.npy", np.asarray(source.labels[retained_rows], dtype=np.int64))
    (output / "ids.json").write_text(json.dumps(retained_ids))
    meta_path = source.directory / "records_meta.jsonl"
    if meta_path.exists():
        retained = []
        for line in meta_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("sample_id") not in forget_ids:
                    retained.append(json.dumps(row, sort_keys=True))
        (output / "records_meta.jsonl").write_text("\n".join(retained) + ("\n" if retained else ""))
    manifest = {"source_directory": str(source.directory), "source_records": len(source.ids), "forgotten_records": len(forget_ids), "retained_records": len(retained_ids), "forgotten_ids_present": any(sample_id in retained_ids for sample_id in forget_ids), "labels_preserved": source.labels is not None}
    (output / "redaction_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class RedactedTokenStore:
    def __init__(self, base: TokenStore, forget_ids: set[str]):
        self.base = base; self.forget_ids = set(forget_ids); self.ids = [sample_id for sample_id in base.ids if sample_id not in self.forget_ids]; self.sequence_length = base.sequence_length; self.dummy_token_id = base.dummy_token_id

    def get_batch(self, sample_ids: Iterable[str], forget_ids: set[str] | None = None, policy: str = "none") -> TokenBatch:
        requested_forget = forget_ids or set()
        if requested_forget and requested_forget != self.forget_ids:
            raise ValueError("redacted store forget set does not match replay request")
        if policy == "none":
            policy = "filter"
        return self.base.get_batch(sample_ids, self.forget_ids, policy)
