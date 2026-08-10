from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from .dataset import TokenStore
from .losses import causal_lm_sum_loss


@dataclass
class LossAudit:
    mean_token_loss: float
    perplexity: float
    examples: int
    tokens: int


def evaluate_loss(
    model: torch.nn.Module,
    store: TokenStore,
    sample_ids: Iterable[str],
    device: str | torch.device,
    batch_size: int = 4,
    max_examples: int | None = None,
) -> LossAudit:
    ids = list(sample_ids)
    if max_examples is not None:
        ids = ids[:max_examples]
    target_device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = store.get_batch(ids[start : start + batch_size], policy="none")
            batch.input_ids = batch.input_ids.to(target_device)
            batch.attention_mask = batch.attention_mask.to(target_device)
            batch.sample_weights = batch.sample_weights.to(target_device)
            loss, tokens = causal_lm_sum_loss(model, batch)
            total_loss += float(loss.item())
            total_tokens += tokens
    mean = total_loss / max(1, total_tokens)
    return LossAudit(mean_token_loss=mean, perplexity=math.exp(min(20.0, mean)), examples=len(ids), tokens=total_tokens)


def rank_auc(member_scores: list[float], nonmember_scores: list[float]) -> float:
    pairs = [(score, 1) for score in member_scores] + [(score, 0) for score in nonmember_scores]
    if not member_scores or not nonmember_scores:
        return float("nan")
    pairs.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    member_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    m = len(member_scores)
    n = len(nonmember_scores)
    return (member_rank_sum - m * (m + 1) / 2.0) / (m * n)
