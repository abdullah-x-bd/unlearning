from __future__ import annotations

import json
import random
from pathlib import Path

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


def load_records_meta(train_dir: str | Path) -> list[dict]:
    path = Path(train_dir) / "records_meta.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def content_closure(records_meta: list[dict], seed_ids: set[str]) -> set[str]:
    by_id = {row["sample_id"]: row for row in records_meta}
    digests = {by_id[sample_id]["content_sha256"] for sample_id in seed_ids if sample_id in by_id}
    return {row["sample_id"] for row in records_meta if row.get("content_sha256") in digests}


def near_duplicate_closure(records_meta: list[dict], seed_ids: set[str], max_hamming: int) -> set[str]:
    by_id = {row["sample_id"]: row for row in records_meta}
    seeds = [by_id[sample_id].get("simhash64") for sample_id in seed_ids if sample_id in by_id]
    seeds = [value for value in seeds if value]
    if not seeds:
        raise ValueError("near-duplicate closure requested but simhash64 metadata is unavailable")
    from .duplicates import hamming64

    closed = set(seed_ids)
    for row in records_meta:
        value = row.get("simhash64")
        if value and any(hamming64(value, seed) <= max_hamming for seed in seeds):
            closed.add(row["sample_id"])
    return closed


def select_scenario_forget_ids(
    plan: list[MicrobatchSpec],
    scenario: dict,
    train_dir: str | Path,
    seed: int,
) -> set[str]:
    strategy = scenario["strategy"]
    if strategy == "id_file":
        path = Path(scenario["path"])
        forget_ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        if not forget_ids:
            raise ValueError(f"forget ID file is empty: {path}")
    elif strategy == "canary_group":
        canaries_path = Path(train_dir) / "canaries.json"
        groups = json.loads(canaries_path.read_text())
        requested = scenario.get("groups", [0])
        requested_set = {int(value) for value in requested}
        forget_ids = {
            sample_id
            for item in groups
            if int(item["group"]) in requested_set
            for sample_id in item["sample_ids"]
        }
        if not forget_ids:
            raise ValueError("requested canary groups were not found")
    else:
        forget_ids = select_forget_ids(
            plan,
            fraction=float(scenario["fraction"]),
            strategy=strategy,
            seed=seed,
        )

    metadata = None
    if bool(scenario.get("content_closure", False)):
        metadata = load_records_meta(train_dir)
        if not metadata:
            raise ValueError("content closure requested but records_meta.jsonl is unavailable")
        forget_ids = content_closure(metadata, forget_ids)
    if scenario.get("near_duplicate_hamming") is not None:
        metadata = metadata or load_records_meta(train_dir)
        if not metadata:
            raise ValueError("near-duplicate closure requested but records_meta.jsonl is unavailable")
        forget_ids = near_duplicate_closure(metadata, forget_ids, int(scenario["near_duplicate_hamming"]))
    return forget_ids
