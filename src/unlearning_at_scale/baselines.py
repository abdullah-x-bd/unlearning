from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .dataset import TokenStore
from .determinism import seed_microbatch


@dataclass
class BaselineStats:
    method: str
    steps: int
    wall_seconds: float
    final_objective: float
    final_forget_component: float
    final_retain_component: float

    def to_dict(self) -> dict:
        return asdict(self)


def sequence_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sequence summed NLL and token counts."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(dtype=logits.dtype)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    nll = (token_losses * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return nll, counts


@torch.no_grad()
def precompute_reference_nll(
    model: torch.nn.Module,
    store: TokenStore,
    sample_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    values: dict[str, float] = {}
    for start in range(0, len(sample_ids), batch_size):
        ids = sample_ids[start : start + batch_size]
        batch = store.get_batch(ids, policy="none")
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        nll, _ = sequence_nll(model, input_ids, attention_mask)
        for sample_id, value in zip(ids, nll.detach().cpu().tolist()):
            values[sample_id] = float(value)
    return values


def _sample_batch(rng: random.Random, ids: list[str], batch_size: int) -> list[str]:
    if not ids:
        raise ValueError("cannot sample from an empty ID set")
    if len(ids) >= batch_size:
        return rng.sample(ids, batch_size)
    return [rng.choice(ids) for _ in range(batch_size)]


def run_approximate_baseline(
    method: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: TokenStore,
    forget_ids: list[str],
    retain_ids: list[str],
    device: str | torch.device,
    steps: int,
    batch_size: int,
    seed: int,
    retain_weight: float = 1.0,
    beta: float = 0.1,
    reference_nll: dict[str, float] | None = None,
) -> BaselineStats:
    """Run a controlled approximate-unlearning baseline."""
    normalized = method.lower().replace("-", "_")
    if normalized not in {"ga", "grad_diff", "npo"}:
        raise ValueError(f"unsupported baseline method: {method}")
    if normalized == "npo" and reference_nll is None:
        raise ValueError("NPO requires precomputed reference NLL values")

    target_device = torch.device(device)
    model.to(target_device)
    model.train()
    rng = random.Random(seed)
    start_time = time.perf_counter()
    last_objective = float("nan")
    last_forget = float("nan")
    last_retain = 0.0

    for step in range(steps):
        seed_microbatch(seed + step)
        optimizer.zero_grad(set_to_none=True)
        f_ids = _sample_batch(rng, forget_ids, batch_size)
        f_batch = store.get_batch(f_ids, policy="none")
        f_input = f_batch.input_ids.to(target_device)
        f_mask = f_batch.attention_mask.to(target_device)
        f_nll, f_tokens = sequence_nll(model, f_input, f_mask)

        if normalized in {"ga", "grad_diff"}:
            forget_component = -(f_nll / f_tokens).mean()
        else:
            ref = torch.tensor(
                [reference_nll[sample_id] for sample_id in f_ids],
                device=target_device,
                dtype=f_nll.dtype,
            )
            log_ratio = -f_nll + ref
            forget_component = (-(2.0 / beta) * F.logsigmoid(-beta * log_ratio)).mean()

        retain_component = torch.zeros((), device=target_device, dtype=forget_component.dtype)
        if normalized in {"grad_diff", "npo"} and retain_weight > 0:
            r_ids = _sample_batch(rng, retain_ids, batch_size)
            r_batch = store.get_batch(r_ids, policy="none")
            r_input = r_batch.input_ids.to(target_device)
            r_mask = r_batch.attention_mask.to(target_device)
            r_nll, r_tokens = sequence_nll(model, r_input, r_mask)
            retain_component = (r_nll / r_tokens).mean()

        objective = forget_component + retain_weight * retain_component
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite {method} objective at step {step}")
        objective.backward()
        optimizer.step()

        last_objective = float(objective.detach().cpu().item())
        last_forget = float(forget_component.detach().cpu().item())
        last_retain = float(retain_component.detach().cpu().item())

    return BaselineStats(
        method=normalized,
        steps=steps,
        wall_seconds=time.perf_counter() - start_time,
        final_objective=last_objective,
        final_forget_component=last_forget,
        final_retain_component=last_retain,
    )
