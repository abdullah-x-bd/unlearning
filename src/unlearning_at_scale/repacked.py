from __future__ import annotations

from .plan import MicrobatchSpec, build_plan


def build_repacked_plan(
    original_plan: list[MicrobatchSpec],
    forget_ids: set[str],
    start_optimizer_step: int,
    microbatch_size: int,
    grad_accum_steps: int,
    rng_seed: int,
    peak_lr: float,
    warmup_ratio: float,
    schedule: str,
) -> list[MicrobatchSpec]:
    ordered_ids: list[str] = []
    for spec in original_plan:
        if spec.optimizer_step < start_optimizer_step:
            continue
        ordered_ids.extend(sample_id for sample_id in spec.sample_ids if sample_id not in forget_ids)
    return build_plan(
        ordered_ids,
        microbatch_size=microbatch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=1,
        shuffle_seed=0,
        rng_seed=rng_seed,
        peak_lr=peak_lr,
        warmup_ratio=warmup_ratio,
        schedule=schedule,
        shuffle=False,
    )
