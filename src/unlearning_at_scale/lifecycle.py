from __future__ import annotations

import gc

import torch


def release_phase(*objects) -> None:
    for obj in objects:
        if isinstance(obj, torch.optim.Optimizer):
            obj.state.clear()
            for group in obj.param_groups:
                group["params"].clear()
        elif isinstance(obj, torch.nn.Module):
            obj.zero_grad(set_to_none=True)
            obj.to_empty(device=torch.device("meta"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
