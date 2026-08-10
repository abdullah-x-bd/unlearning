from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .rollback import apply_exact_patch, build_exact_patch
from .state import state_sha256


def _object_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_object_bytes(key) + _object_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_object_bytes(item) for item in value)
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, (int, float, bool)):
        return 8
    return 0


def exact_patch_bytes(patch) -> int:
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in patch.tensor_xor.values())
    scalar_bytes = _object_bytes(patch.scalar_before)
    return tensor_bytes + scalar_bytes


def benchmark_checkpoint_patch(before_path: str | Path, after_path: str | Path) -> dict:
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    after = torch.load(after_path, map_location="cpu", weights_only=False)

    start = time.perf_counter()
    patch = build_exact_patch(before, after)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    restored = apply_exact_patch(after, patch)
    apply_seconds = time.perf_counter() - start

    before_hash = state_sha256(before)
    restored_hash = state_sha256(restored)
    return {
        "exact": before_hash == restored_hash,
        "before_sha256": before_hash,
        "after_sha256": state_sha256(after),
        "restored_sha256": restored_hash,
        "before_checkpoint_bytes": Path(before_path).stat().st_size,
        "after_checkpoint_bytes": Path(after_path).stat().st_size,
        "patch_payload_bytes": exact_patch_bytes(patch),
        "patch_build_seconds": build_seconds,
        "patch_apply_seconds": apply_seconds,
        "tensor_patch_count": len(patch.tensor_xor),
        "scalar_patch_count": len(patch.scalar_before),
    }


def write_benchmark(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
