# Results schema

Each experiment directory is self-describing.

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
    summary.json
    checkpoints/
  forget/<scenario>/
    forget_ids.txt
    oracle/summary.json
    replay-slot_mask/summary.json
    replay-filter/summary.json
    repacked/summary.json
    summary.json
```

The top-level summary is the only file intended for automated aggregation. Raw artifacts remain available for verification.

A release aggregation should emit one row per model, seed, forget scenario, replay policy, dtype, attention backend, and hardware configuration.
