from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from .dataset import TokenStore
from .determinism import seed_microbatch
from .losses import causal_lm_sum_loss
from .plan import MicrobatchSpec
from .state import save_checkpoint
from .wal import WalWriter


@dataclass
class RunStats:
    wall_seconds: float
    logical_steps_seen: int
    applied_updates: int
    skipped_updates: int
    retained_examples: int
    retained_tokens: int
    summed_training_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


def autocast_context(device: torch.device, dtype: str):
    if dtype == "fp32":
        return nullcontext()
    if dtype != "bf16":
        raise ValueError("supported training dtypes are fp32 and bf16")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    foreach: bool = False,
    fused: bool = False,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        foreach=foreach,
        fused=fused,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _gpu_memory_text(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f" | gpu {allocated:.2f}G alloc {reserved:.2f}G reserved {peak:.2f}G peak"


class TraceRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        token_store: TokenStore,
        device: str | torch.device,
        dtype: str = "fp32",
        loss_fn: Callable = causal_lm_sum_loss,
    ):
        self.model = model
        self.optimizer = optimizer
        self.token_store = token_store
        self.device = torch.device(device)
        self.dtype = dtype
        self.loss_fn = loss_fn
        self.model.to(self.device)

    def run(
        self,
        plan: Iterable[MicrobatchSpec],
        forget_ids: set[str] | None = None,
        policy: str = "none",
        wal_writer: WalWriter | None = None,
        start_optimizer_step: int = 0,
        checkpoint_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        progress_every: int | None = 50,
        progress_label: str | None = None,
    ) -> RunStats:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        forget = forget_ids or set()
        segment_has_data = False
        start = time.perf_counter()
        applied_updates = 0
        skipped_updates = 0
        logical_steps_seen: set[int] = set()
        retained_examples = 0
        retained_tokens = 0
        summed_loss = 0.0

        plan_items = list(plan)
        total_updates = len(
            {
                spec.optimizer_step
                for spec in plan_items
                if spec.optimizer_step >= start_optimizer_step
            }
        )
        completed_updates = 0
        label = progress_label or policy
        if progress_every and total_updates:
            print(
                f"[{label}] starting {total_updates} optimizer updates from logical step "
                f"{start_optimizer_step}",
                flush=True,
            )

        for spec in plan_items:
            if spec.optimizer_step < start_optimizer_step:
                continue
            logical_steps_seen.add(spec.optimizer_step)
            if wal_writer is not None:
                wal_writer.append(spec)

            seed_microbatch(spec.seed)
            batch = self.token_store.get_batch(spec.sample_ids, forget, policy)
            if batch.retained_count > 0:
                batch = batch.to(self.device)
                for group in self.optimizer.param_groups:
                    group["lr"] = float(spec.lr)
                with autocast_context(self.device, self.dtype):
                    loss, token_count = self.loss_fn(self.model, batch)
                loss.backward()
                segment_has_data = True
                retained_examples += batch.retained_count
                retained_tokens += token_count
                summed_loss += float(loss.detach().cpu().item())

            if spec.accumulation_end:
                if segment_has_data:
                    self.optimizer.step()
                    applied_updates += 1
                else:
                    skipped_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
                segment_has_data = False
                completed_updates += 1

                next_step = spec.optimizer_step + 1
                if (
                    checkpoint_every
                    and checkpoint_dir is not None
                    and next_step % checkpoint_every == 0
                ):
                    save_checkpoint(
                        Path(checkpoint_dir) / f"step-{next_step:06d}.pt",
                        self.model,
                        self.optimizer,
                        next_optimizer_step=next_step,
                    )

                if progress_every and (
                    completed_updates % progress_every == 0
                    or completed_updates == total_updates
                ):
                    elapsed = time.perf_counter() - start
                    rate = completed_updates / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, total_updates - completed_updates)
                    eta = remaining / rate if rate > 0 else 0.0
                    percent = 100.0 * completed_updates / total_updates
                    mean_token_loss = summed_loss / retained_tokens if retained_tokens else 0.0
                    print(
                        f"[{label}] update {completed_updates}/{total_updates} "
                        f"({percent:.1f}%) | elapsed {_format_duration(elapsed)} "
                        f"| eta {_format_duration(eta)} | mean token loss {mean_token_loss:.6f}"
                        f"{_gpu_memory_text(self.device)}",
                        flush=True,
                    )

        return RunStats(
            wall_seconds=time.perf_counter() - start,
            logical_steps_seen=len(logical_steps_seen),
            applied_updates=applied_updates,
            skipped_updates=skipped_updates,
            retained_examples=retained_examples,
            retained_tokens=retained_tokens,
            summed_training_loss=summed_loss,
        )
