from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    # Flatten first so zero-dimensional optimizer scalars such as AdamW's
    # step tensor can be reinterpreted as bytes on every supported dtype.
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _update_hash(hasher: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(json.dumps(list(tensor.shape)).encode())
        hasher.update(_tensor_bytes(tensor))
        return
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value.keys(), key=lambda item: repr(item)):
            hasher.update(repr(key).encode())
            _update_hash(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode())
        for item in value:
            _update_hash(hasher, item)
        return
    hasher.update(type(value).__name__.encode())
    hasher.update(repr(value).encode())


def state_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, value)
    return hasher.hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    return state_sha256(model.state_dict())


def optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    return state_sha256(optimizer.state_dict())


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_optimizer_step: int,
    metadata: dict | None = None,
) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_optimizer_step": int(next_optimizer_step),
        "metadata": metadata or {},
    }
    torch.save(payload, target)
    return {
        "path": str(target),
        "next_optimizer_step": int(next_optimizer_step),
        "model_sha256": state_sha256(payload["model"]),
        "optimizer_sha256": state_sha256(payload["optimizer"]),
        "bytes": target.stat().st_size,
    }


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return payload
