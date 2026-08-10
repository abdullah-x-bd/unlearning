from __future__ import annotations

import random
from .plan import MicrobatchSpec


def first_occurrence_steps(plan: list[MicrobatchSpec]) -> dict[str, int]:
    first: dict[str, int] = {}
    for spec in plan:
        for sample_id in spec.sample_ids:
            first.setdefault(sample_id, spec.optimizer_step)
    return first


def select_forget_ids(
    plan: list[MicrobatchSpec],
    fraction: float,
    strategy: str,
    seed: int,
) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("forget fraction must be between 0 and 1")
    first = first_occurrence_steps(plan)
    ids = list(first)
    count = max(1, round(len(ids) * fraction))
    rng = random.Random(seed)

    if strategy == "random":
        return set(rng.sample(ids, min(count, len(ids))))

    ordered = sorted(ids, key=lambda sample_id: (first[sample_id], sample_id))
    if strategy == "early":
        pool = ordered[: max(count, len(ordered) // 3)]
    elif strategy == "middle":
        start = len(ordered) // 3
        end = max(start + count, 2 * len(ordered) // 3)
        pool = ordered[start:end]
    elif strategy == "late":
        pool = ordered[2 * len(ordered) // 3 :]
    else:
        raise ValueError(f"unknown forget strategy: {strategy}")
    return set(rng.sample(pool, min(count, len(pool))))


def earliest_forget_step(plan: list[MicrobatchSpec], forget_ids: set[str]) -> int:
    first = first_occurrence_steps(plan)
    affected = [first[sample_id] for sample_id in forget_ids if sample_id in first]
    if not affected:
        raise ValueError("forget set does not occur in the execution plan")
    return min(affected)


def content_closure(records_meta: list[dict], seed_ids: set[str]) -> set[str]:
    by_id = {row["sample_id"]: row for row in records_meta}
    digests = {by_id[sample_id]["content_sha256"] for sample_id in seed_ids if sample_id in by_id}
    return {row["sample_id"] for row in records_meta if row.get("content_sha256") in digests}
