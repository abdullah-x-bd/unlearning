from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
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
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes differ")
        if self.input_ids.shape[0] != len(self.ids):
            raise ValueError("ids.json does not match token arrays")
        self.sequence_length = int(self.input_ids.shape[1])
        self.dummy_token_id = int(dummy_token_id)

    def __len__(self) -> int:
        return len(self.ids)

    def get_batch(
        self,
        sample_ids: Iterable[str],
        forget_ids: set[str] | None = None,
        policy: str = "none",
    ) -> TokenBatch:
        forget = forget_ids or set()
        ordered = tuple(sample_ids)
        if policy not in {"none", "filter", "slot_mask"}:
            raise ValueError(f"unknown replay policy: {policy}")

        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        weights: list[float] = []
        output_ids: list[str] = []

        for sample_id in ordered:
            is_forgotten = sample_id in forget
            if is_forgotten and policy == "filter":
                continue

            if is_forgotten and policy == "slot_mask":
                rows.append(np.full((self.sequence_length,), self.dummy_token_id, dtype=np.int64))
                masks.append(np.ones((self.sequence_length,), dtype=np.int64))
                weights.append(0.0)
                output_ids.append(sample_id)
                continue

            row = self.id_to_row.get(sample_id)
            if row is None:
                raise KeyError(f"sample ID not found in token store: {sample_id}")
            rows.append(np.asarray(self.input_ids[row], dtype=np.int64))
            masks.append(np.asarray(self.attention_mask[row], dtype=np.int64))
            weights.append(1.0)
            output_ids.append(sample_id)

        if not rows:
            empty = torch.empty((0, self.sequence_length), dtype=torch.long)
            return TokenBatch(
                input_ids=empty,
                attention_mask=empty.clone(),
                sample_weights=torch.empty((0,), dtype=torch.float32),
                sample_ids=tuple(),
                retained_count=0,
            )

        return TokenBatch(
            input_ids=torch.from_numpy(np.stack(rows, axis=0).copy()).long(),
            attention_mask=torch.from_numpy(np.stack(masks, axis=0).copy()).long(),
            sample_weights=torch.tensor(weights, dtype=torch.float32),
            sample_ids=tuple(output_ids),
            retained_count=sum(weight > 0 for weight in weights),
        )

    def redact(self, forget_ids: set[str]) -> "RedactedTokenStore":
        return RedactedTokenStore(self, forget_ids)


class RedactedTokenStore:
    def __init__(self, base: TokenStore, forget_ids: set[str]):
        self.base = base
        self.forget_ids = set(forget_ids)
        self.ids = [sample_id for sample_id in base.ids if sample_id not in self.forget_ids]
        self.sequence_length = base.sequence_length
        self.dummy_token_id = base.dummy_token_id

    def get_batch(self, sample_ids: Iterable[str], policy: str) -> TokenBatch:
        if policy == "none":
            policy = "filter"
        return self.base.get_batch(sample_ids, self.forget_ids, policy)
