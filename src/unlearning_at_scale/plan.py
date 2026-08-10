from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MicrobatchSpec:
    index: int
    sample_ids: tuple[str, ...]
    seed: int
    lr: float
    optimizer_step: int
    accumulation_end: bool

    def to_json(self) -> dict:
        out = asdict(self)
        out["sample_ids"] = list(self.sample_ids)
        return out

    @classmethod
    def from_json(cls, payload: dict) -> "MicrobatchSpec":
        return cls(
            index=int(payload["index"]),
            sample_ids=tuple(payload["sample_ids"]),
            seed=int(payload["seed"]),
            lr=float(payload["lr"]),
            optimizer_step=int(payload["optimizer_step"]),
            accumulation_end=bool(payload["accumulation_end"]),
        )


def _derive_seed(base_seed: int, microbatch_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{microbatch_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _quantize_lr(value: float) -> float:
    return float(np.float32(value))


def scheduled_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_ratio: float,
    schedule: str,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0
    if warmup_steps and step < warmup_steps:
        return _quantize_lr(peak_lr * float(step + 1) / float(warmup_steps))

    if schedule == "constant":
        return _quantize_lr(peak_lr)
    if schedule != "cosine":
        raise ValueError(f"unsupported schedule: {schedule}")

    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    value = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return _quantize_lr(value)


def build_plan(
    sample_ids: Iterable[str],
    microbatch_size: int,
    grad_accum_steps: int,
    epochs: int,
    shuffle_seed: int,
    rng_seed: int,
    peak_lr: float,
    warmup_ratio: float = 0.0,
    schedule: str = "constant",
    shuffle: bool = True,
) -> list[MicrobatchSpec]:
    if microbatch_size < 1 or grad_accum_steps < 1 or epochs < 1:
        raise ValueError("microbatch_size, grad_accum_steps, and epochs must be positive")

    base_ids = list(sample_ids)
    if not base_ids:
        raise ValueError("sample_ids is empty")

    epoch_orders: list[list[str]] = []
    for epoch in range(epochs):
        order = list(base_ids)
        if shuffle:
            rng = np.random.default_rng(shuffle_seed + epoch)
            rng.shuffle(order)
        epoch_orders.append(order)

    total_microbatches = sum(math.ceil(len(order) / microbatch_size) for order in epoch_orders)
    total_steps = math.ceil(total_microbatches / grad_accum_steps)

    records: list[MicrobatchSpec] = []
    mb_index = 0
    logical_step = 0
    in_accum = 0
    for order in epoch_orders:
        for start in range(0, len(order), microbatch_size):
            batch_ids = tuple(order[start : start + microbatch_size])
            in_accum += 1
            is_last_overall = mb_index == total_microbatches - 1
            accum_end = in_accum == grad_accum_steps or is_last_overall
            lr = scheduled_lr(logical_step, total_steps, peak_lr, warmup_ratio, schedule)
            records.append(
                MicrobatchSpec(
                    index=mb_index,
                    sample_ids=batch_ids,
                    seed=_derive_seed(rng_seed, mb_index),
                    lr=lr,
                    optimizer_step=logical_step,
                    accumulation_end=accum_end,
                )
            )
            mb_index += 1
            if accum_end:
                logical_step += 1
                in_accum = 0
    return records


def write_plan(path: str | Path, records: Iterable[MicrobatchSpec]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with target.open("wb") as handle:
        for record in records:
            line = (json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            hasher.update(line)
    return hasher.hexdigest()


def read_plan(path: str | Path) -> list[MicrobatchSpec]:
    records: list[MicrobatchSpec] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(MicrobatchSpec.from_json(json.loads(line)))
    return records


def plan_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
