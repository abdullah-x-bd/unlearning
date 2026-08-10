from __future__ import annotations

import math
from typing import Iterable

import torch

from .dataset import TokenStore
from .losses import causal_lm_sum_loss


def _zero_like_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}


def diagonal_fisher(
    model: torch.nn.Module,
    store: TokenStore,
    sample_ids: Iterable[str],
    device: str | torch.device,
    max_examples: int = 32,
) -> dict[str, torch.Tensor]:
    target_device = torch.device(device)
    fisher = _zero_like_trainable(model)
    ids = list(sample_ids)[:max_examples]
    model.train()
    for sample_id in ids:
        model.zero_grad(set_to_none=True)
        batch = store.get_batch([sample_id], policy="none")
        batch.input_ids = batch.input_ids.to(target_device)
        batch.attention_mask = batch.attention_mask.to(target_device)
        batch.sample_weights = batch.sample_weights.to(target_device)
        loss, _ = causal_lm_sum_loss(model, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name].add_(param.grad.detach().pow(2))
    scale = 1.0 / max(1, len(ids))
    for value in fisher.values():
        value.mul_(scale)
    return fisher


def curvature_anti_update(
    model: torch.nn.Module,
    store: TokenStore,
    forget_ids: Iterable[str],
    fisher: dict[str, torch.Tensor],
    device: str | torch.device,
    step_size: float,
    damping: float,
    trust_radius: float,
) -> float:
    ids = list(forget_ids)
    target_device = torch.device(device)
    model.zero_grad(set_to_none=True)
    for sample_id in ids:
        batch = store.get_batch([sample_id], policy="none")
        batch.input_ids = batch.input_ids.to(target_device)
        batch.attention_mask = batch.attention_mask.to(target_device)
        batch.sample_weights = batch.sample_weights.to(target_device)
        loss, _ = causal_lm_sum_loss(model, batch)
        loss.backward()

    updates: dict[str, torch.Tensor] = {}
    norm_sq = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        update = step_size * param.grad / (fisher[name].to(param.device) + damping)
        updates[name] = update
        norm_sq += float(update.detach().float().pow(2).sum().item())

    norm = math.sqrt(norm_sq)
    scale = min(1.0, trust_radius / max(norm, 1e-12))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in updates:
                param.add_(updates[name], alpha=scale)
    model.zero_grad(set_to_none=True)
    return norm * scale
