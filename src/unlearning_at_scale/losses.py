from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX, TokenBatch


def causal_lm_sum_loss(model: torch.nn.Module, batch: TokenBatch) -> tuple[torch.Tensor, int]:
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    valid_targets = shift_labels.ne(IGNORE_INDEX)
    attention = batch.attention_mask[:, 1:].bool()
    token_mask = (valid_targets & attention).to(dtype=logits.dtype)
    token_mask = token_mask * batch.sample_weights.to(dtype=logits.dtype).unsqueeze(1)
    safe_labels = shift_labels.masked_fill(~valid_targets, 0)
    losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), safe_labels.view(-1), reduction="none").view_as(shift_labels)
    total = (losses * token_mask).sum()
    token_count = int(token_mask.detach().sum().item())
    return total, token_count
