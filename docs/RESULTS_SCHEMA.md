# Results schema

Each experiment directory is self-describing and designed for machine aggregation.

```text
runs/<experiment>/
  config.resolved.json
  environment.json
  execution_plan.jsonl
  summary.json
  original/
    trace.wal
    trace.wal.sha256
    manifest.jsonl
    manifest.jsonl.sha256
    final-model-state.pt
    summary.json
    checkpoints/
  replay-identity/
    summary.json
  forget/<scenario>/
    forget_ids.txt
    redacted-data/
    oracle/
      final-model-state.pt
      summary.json
    replay-slot_mask/
      summary.json
    replay-filter/
      summary.json
    repacked/
      summary.json
    summary.json
```

Additional studies can write:

```text
provenance-ablations/provenance_ablations.json
approximate-baselines/<method>/summary.json
checkpoint-sweep.json
rollback-benchmark.json
lora-cohort/summary.json
hotpath/summary.json
```

## Required exactness fields

An exact comparison row should preserve:

- left SHA-256
- right SHA-256
- exact boolean
- total tensors
- unequal tensors
- total elements
- unequal elements
- maximum absolute difference
- L2 difference
- optimizer-state hash comparison

## Required systems fields

Where applicable preserve:

- wall-clock seconds
- replay distance in logical steps
- WAL bytes
- manifest bytes
- total provenance bytes
- checkpoint bytes
- retained checkpoint bytes
- patch bytes
- adapter bytes

## Aggregation

`scripts/aggregate_results.py` scans `summary.json` artifacts and emits JSON plus CSV for downstream tables and plots. Raw artifacts remain the source of truth.

A paper table row should always include enough keys to identify:

- dataset
- model
- immutable revision
- seed
- hardware
- dtype
- attention backend
- deletion scenario
- replay policy or baseline method
