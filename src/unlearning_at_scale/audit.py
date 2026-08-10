from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .dataset import TokenStore
from .losses import causal_lm_sum_loss


@dataclass
class LossAudit:
    mean_token_loss: float
    perplexity: float
    examples: int
    tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CanaryAudit:
    groups: int
    exact_greedy_matches: int
    exact_greedy_rate: float
    mean_completion_nll: float
    mean_completion_perplexity: float

    def to_dict(self) -> dict:
        return asdict(self)


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


def per_example_token_losses(
    model: torch.nn.Module,
    store: TokenStore,
    sample_ids: Iterable[str],
    device: str | torch.device,
    max_examples: int | None = None,
) -> list[float]:
    ids = list(sample_ids)
    if max_examples is not None:
        ids = ids[:max_examples]
    target_device = torch.device(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for sample_id in ids:
            batch = store.get_batch([sample_id], policy="none")
            batch.input_ids = batch.input_ids.to(target_device)
            batch.attention_mask = batch.attention_mask.to(target_device)
            batch.sample_weights = batch.sample_weights.to(target_device)
            loss, tokens = causal_lm_sum_loss(model, batch)
            scores.append(float(loss.item()) / max(1, tokens))
    return scores


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


def loss_membership_auc(
    model: torch.nn.Module,
    member_store: TokenStore,
    member_ids: Iterable[str],
    nonmember_store: TokenStore,
    nonmember_ids: Iterable[str],
    device: str | torch.device,
    max_examples: int = 128,
) -> dict:
    member_losses = per_example_token_losses(model, member_store, member_ids, device, max_examples)
    nonmember_losses = per_example_token_losses(model, nonmember_store, nonmember_ids, device, max_examples)
    auc = rank_auc([-value for value in member_losses], [-value for value in nonmember_losses])
    return {
        "auc": auc,
        "member_examples": len(member_losses),
        "nonmember_examples": len(nonmember_losses),
        "member_mean_loss": sum(member_losses) / max(1, len(member_losses)),
        "nonmember_mean_loss": sum(nonmember_losses) / max(1, len(nonmember_losses)),
    }


def _completion_nll(model, tokenizer, prompt: str, completion: str, device: torch.device) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=full_ids, use_cache=False).logits[:, :-1, :]
    targets = full_ids[:, 1:]
    start = max(0, prompt_len - 1)
    logits = logits[:, start:, :]
    targets = targets[:, start:]
    if targets.numel() == 0:
        return 0.0, 0
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
    return float(loss.item()), int(targets.numel())


def evaluate_canaries(
    model: torch.nn.Module,
    tokenizer,
    canaries_path: str | Path,
    device: str | torch.device,
    max_groups: int | None = None,
) -> CanaryAudit:
    canaries = json.loads(Path(canaries_path).read_text())
    if max_groups is not None:
        canaries = canaries[:max_groups]
    target_device = torch.device(device)
    model.eval()
    exact = 0
    total_nll = 0.0
    total_tokens = 0
    for item in canaries:
        prompt = item["prompt"]
        completion = item["completion"]
        nll, tokens = _completion_nll(model, tokenizer, prompt, completion, target_device)
        total_nll += nll
        total_tokens += tokens
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(target_device)
        expected = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if expected:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=prompt_tokens,
                    max_new_tokens=len(expected),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                )
            continuation = generated[0, prompt_tokens.shape[1] :].tolist()
            exact += int(continuation == expected)
    mean_nll = total_nll / max(1, total_tokens)
    return CanaryAudit(
        groups=len(canaries),
        exact_greedy_matches=exact,
        exact_greedy_rate=exact / max(1, len(canaries)),
        mean_completion_nll=mean_nll,
        mean_completion_perplexity=math.exp(min(20.0, mean_nll)),
    )
