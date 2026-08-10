from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


PathKey = tuple[tuple[str, Any], ...]


@dataclass
class ExactPatch:
    tensor_xor: dict[PathKey, torch.Tensor]
    scalar_before: dict[PathKey, Any]


def _walk(value: Any, path: PathKey = ()):
    if torch.is_tensor(value):
        yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, path + (("dict", key),))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (("list", index),))
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk(item, path + (("tuple", index),))
        return
    yield path, value


def _get(root: Any, path: PathKey) -> Any:
    value = root
    for kind, key in path:
        value = value[key]
    return value


def _set(root: Any, path: PathKey, value: Any) -> Any:
    if not path:
        return value
    parent = root
    for kind, key in path[:-1]:
        parent = parent[key]
    kind, key = path[-1]
    if kind == "tuple":
        raise TypeError("tuple leaf replacement is not supported")
    parent[key] = value
    return root


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().view(torch.uint8).clone()


def build_exact_patch(before: Any, after: Any) -> ExactPatch:
    before_items = dict(_walk(before))
    after_items = dict(_walk(after))
    if before_items.keys() != after_items.keys():
        raise ValueError("state structures differ")

    tensor_xor: dict[PathKey, torch.Tensor] = {}
    scalar_before: dict[PathKey, Any] = {}
    for path in before_items:
        left = before_items[path]
        right = after_items[path]
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"tensor metadata differs at {path}")
            tensor_xor[path] = torch.bitwise_xor(_tensor_bytes(left), _tensor_bytes(right))
        elif left != right:
            scalar_before[path] = copy.deepcopy(left)
    return ExactPatch(tensor_xor=tensor_xor, scalar_before=scalar_before)


def apply_exact_patch(after: Any, patch: ExactPatch) -> Any:
    restored = copy.deepcopy(after)
    for path, xor_bytes in patch.tensor_xor.items():
        current = _get(restored, path)
        current_cpu = current.detach().cpu().contiguous()
        restored_bytes = torch.bitwise_xor(current_cpu.view(torch.uint8), xor_bytes)
        restored_tensor = restored_bytes.view(current_cpu.dtype).reshape(current_cpu.shape).clone()
        restored = _set(restored, path, restored_tensor)
    for path, value in patch.scalar_before.items():
        restored = _set(restored, path, copy.deepcopy(value))
    return restored
