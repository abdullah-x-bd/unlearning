from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .state import state_sha256


@dataclass
class TensorComparison:
    exact: bool
    total_tensors: int
    unequal_tensors: int
    unequal_elements: int
    total_elements: int
    max_abs_diff: float
    l2_diff: float
    left_sha256: str
    right_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_state_dicts(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> TensorComparison:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise ValueError(f"state dict keys differ: {missing[:10]}")

    unequal_tensors = 0
    unequal_elements = 0
    total_elements = 0
    max_abs = 0.0
    l2_sq = 0.0

    for key in sorted(left):
        a = left[key].detach().cpu()
        b = right[key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"tensor metadata differs for {key}")
        total_elements += a.numel()
        equal = torch.equal(a, b)
        if not equal:
            unequal_tensors += 1
            diff_mask = a != b
            unequal_elements += int(diff_mask.sum().item())
            if a.is_floating_point() or a.is_complex():
                diff = (a.to(torch.float64) - b.to(torch.float64)).abs()
                if diff.numel():
                    max_abs = max(max_abs, float(diff.max().item()))
                    l2_sq += float((diff * diff).sum().item())
            else:
                max_abs = max(max_abs, 1.0)
                l2_sq += float(diff_mask.sum().item())

    return TensorComparison(
        exact=unequal_tensors == 0,
        total_tensors=len(left),
        unequal_tensors=unequal_tensors,
        unequal_elements=unequal_elements,
        total_elements=total_elements,
        max_abs_diff=max_abs,
        l2_diff=l2_sq**0.5,
        left_sha256=state_sha256(left),
        right_sha256=state_sha256(right),
    )
