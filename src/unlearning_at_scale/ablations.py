from __future__ import annotations

from dataclasses import replace

from .plan import MicrobatchSpec


def ablate_plan(plan: list[MicrobatchSpec], ablation: str) -> list[MicrobatchSpec]:
    """Return a deliberately corrupted execution plan for provenance sufficiency tests.

    Each ablation changes one recorded control variable while leaving the others fixed.
    The resulting divergence is empirical evidence about which provenance fields matter.
    """
    if not plan:
        raise ValueError("plan is empty")

    out = list(plan)
    if ablation == "seed":
        return [replace(spec, seed=spec.seed ^ 0x5A5A5A5A) for spec in out]

    if ablation == "learning_rate":
        return [replace(spec, lr=float(spec.lr) * 1.01) for spec in out]

    if ablation == "sample_order":
        changed: list[MicrobatchSpec] = []
        for spec in out:
            ids = tuple(reversed(spec.sample_ids))
            changed.append(replace(spec, sample_ids=ids))
        return changed

    if ablation == "microbatch_assignment":
        flat = [sample_id for spec in out for sample_id in spec.sample_ids]
        if len(flat) < 2:
            raise ValueError("microbatch assignment ablation needs at least two samples")
        flat = flat[1:] + flat[:1]
        cursor = 0
        changed = []
        for spec in out:
            size = len(spec.sample_ids)
            changed.append(replace(spec, sample_ids=tuple(flat[cursor : cursor + size])))
            cursor += size
        return changed

    if ablation == "accumulation_boundary":
        if len(out) < 2:
            raise ValueError("accumulation ablation needs at least two microbatches")
        changed = []
        for index, spec in enumerate(out):
            if index == 0:
                changed.append(replace(spec, accumulation_end=True))
            elif index == 1 and out[0].optimizer_step == spec.optimizer_step:
                changed.append(replace(spec, accumulation_end=False, optimizer_step=spec.optimizer_step + 1))
            else:
                changed.append(spec)
        return changed

    if ablation == "optimizer_step":
        return [replace(spec, optimizer_step=spec.optimizer_step + 1) for spec in out]

    raise ValueError(f"unknown provenance ablation: {ablation}")


def supported_ablations() -> tuple[str, ...]:
    return (
        "seed",
        "learning_rate",
        "sample_order",
        "microbatch_assignment",
        "accumulation_boundary",
        "optimizer_step",
    )
