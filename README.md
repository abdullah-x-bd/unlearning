# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models

[![CI](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml)

This repository is the experimental artifact for a ground-up reconstruction of **Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models**.

The paper is intentionally downstream of the artifact. The old manuscript is treated as a source of hypotheses, not as evidence. Claims for the rebuilt manuscript will be written from released experiment outputs after the full GPU program is complete.

## Research question

Can language-model training be engineered so that a later data-deletion request becomes a reproducible counterfactual computation rather than an ad hoc post-training edit?

The exact target used here is a **trace-preserving deletion counterfactual**. A training execution plan fixes microbatch slots, random seeds, optimizer-step boundaries, and learning-rate values. When a deletion request arrives, the requested records contribute zero loss while the original execution plan remains fixed.

This is deliberately separated from ordinary retraining on a densely repacked `D \ F` dataset. Repacking can change batch composition, random-number consumption, floating-point reduction order, optimizer-step boundaries, and the learning-rate trajectory. Both counterfactuals are measured rather than conflated.

## Experimental principles

1. Exactness is measured, never assumed.
2. Failures of deterministic replay are results.
3. Byte equality is scoped to a recorded software and hardware environment.
4. Model state and optimizer state are both hashed.
5. The 32-byte WAL claim refers only to the fixed-width binary record. Ordered sample IDs are stored separately and their cost is reported.
6. Main replay experiments can physically remove forgotten rows from the token store before replay.
7. Approximate methods are labeled approximate and evaluated separately.
8. Distributed exactness is not claimed until a dedicated multi-GPU study exists.
9. Release runs must pin immutable model revisions.
10. The paper is rewritten only after the claims ledger is backed by released artifacts.

## What is implemented

### Exact replay core

- immutable microbatch execution plans
- deterministic per-microbatch RNG seeds
- fixed accumulation boundaries
- float32 learning-rate values recorded in the trace
- 32-byte WAL records with CRC32
- segment SHA-256 verification
- ordered-ID manifest with SHA-256 or HMAC-SHA256 digests
- exact model-state hashing
- exact optimizer-state hashing
- checkpoint restore and replay
- independent trace-preserving deletion oracle
- slot-preserving masked replay
- physical row filtering replay
- repacked retain-only retraining baseline
- physically materialized redacted token stores

### Experimental ablations

- RNG seed ablation
- learning-rate ablation
- sample-order ablation
- microbatch-assignment ablation
- accumulation-boundary ablation
- logical-step ablation
- FP32 versus BF16
- eager versus SDPA attention
- dropout enabled versus disabled
- deterministic algorithms enabled versus relaxed
- optimizer foreach and fused variants
- checkpoint-distance sweeps

### Deletion geometry

- early deletion
- middle deletion
- late deletion
- randomly distributed deletion
- exact-content duplicate closure
- SimHash near-duplicate closure
- synthetic canary-group deletion
- official TOFU forget01, forget05, and forget10 ID sets

### Approximate comparison methods

- gradient ascent
- gradient difference with retain regularization
- negative preference optimization
- diagonal-Fisher curvature anti-update

### Operational extensions

- exact XOR byte rollback patches
- cohort-scoped LoRA training and exact base-recovery check after adapter unload
- checkpoint storage versus replay-latency benchmarking

### Audits

- forget-set token loss
- retain-set token loss
- held-out perplexity
- loss-based membership inference AUC
- canary completion NLL
- canary exact greedy extraction rate
- exact tensor divergence from the deletion oracle

## Model scaling matrix

The primary scaling study uses one architecture family so model size changes without simultaneously changing the overall model family:

- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b`, budget-dependent extension

Development configs currently use `revision: main`. Final release configs must replace floating revisions with immutable commit SHAs and set `release_mode: true`.

## Datasets

### WikiText-103 trace study

The primary systems study converts WikiText-103 into fixed-length causal-LM records and adds controlled synthetic canary groups.

```bash
python -m pip install -e '.[llm,dev]'
python scripts/prepare_dataset.py \
  --dataset Salesforce/wikitext \
  --subset wikitext-103-raw-v1 \
  --model EleutherAI/pythia-160m \
  --output data/prepared/wikitext103-pythia-256 \
  --max-records 20000 \
  --sequence-length 256
```

The preparation artifact includes immutable sample IDs, content SHA-256 values, SimHash signatures, token arrays, canary metadata, and a dataset manifest.

### TOFU benchmark track

A second preparation path converts the official TOFU full set into the same traceable token-store format and maps the official forget and retain splits to immutable IDs.

```bash
python scripts/prepare_tofu.py \
  --model EleutherAI/pythia-1.4b \
  --output data/prepared/tofu-pythia-1.4b-256 \
  --sequence-length 256
```

This produces `forget01_ids.txt`, `forget05_ids.txt`, `forget10_ids.txt`, and the corresponding retain lists.

## Replay semantics

### `slot_mask`

The original microbatch shape is retained. A forgotten sample ID is replaced by a deterministic dummy sequence and its token-loss weight is set to zero. No forgotten token row is required.

When `materialize_redacted_store: true`, the experiment writes a new token store that physically omits the forgotten rows and then runs replay against that redacted store.

### `filter`

Forgotten rows are removed from the microbatch before the forward pass. This is mathematically natural under a sum-reduced objective but changes batch shapes, so exactness is tested rather than assumed.

### `repacked`

Retained examples are densely repacked into new microbatches and a new schedule is constructed. This represents ordinary retain-set retraining from the selected checkpoint. It is a different counterfactual from trace-preserving replay and is measured as such.

## Fixed-width WAL

Each binary record uses:

```text
uint64 ordered_sample_ids_digest
uint64 microbatch_seed
float32 learning_rate
uint32 logical_optimizer_step
uint16 microbatch_length
uint16 flags
uint32 crc32
```

The binary record is exactly **32 bytes per microbatch**.

The ordered IDs are held in a separate manifest. Every run reports:

- WAL bytes
- manifest bytes
- total provenance bytes
- provenance bytes per microbatch
- WAL SHA-256
- manifest SHA-256

Set `UNLEARNING_WAL_HMAC_KEY` to use HMAC-SHA256 for ordered-ID digests rather than public SHA-256-derived digests.

## Exactness evidence

Every exact comparison records:

- full model-state SHA-256
- full optimizer-state SHA-256
- exact-hash equality
- total tensor count
- unequal tensor count
- unequal element count
- maximum absolute parameter difference
- L2 parameter difference
- applied optimizer updates
- skipped logical updates
- wall-clock time
- checkpoint size
- replay distance

A byte-exact claim is eligible only when both the hash and tensor comparison agree.

## Run the dependency-light regression suite

```bash
python -m pip install -e '.[dev]'
pytest
uas core-smoke --output runs/core-smoke
```

The CI model is intentionally tiny and local. It tests software invariants only and is never used as scientific evidence.

## Run the main scale study

```bash
uas run configs/pythia-160m.yaml
python scripts/run_matrix.py configs/matrix-main.yaml
```

The scale matrix uses a canonical late 1 percent deletion across model sizes. Deletion geometry is studied separately on 410M:

```bash
uas run configs/pythia-410m-deletion-geometry.yaml
uas run configs/pythia-410m-canary.yaml
```

The separate design avoids multiplying every expensive model size by every deletion geometry.

## Run the TOFU track

```bash
uas run configs/tofu-pythia-1.4b.yaml
```

This is a separate benchmark-facing study and must not be mixed with WikiText rows in the same result table without an explicit dataset column.

## Run determinism stress tests

```bash
python scripts/run_stress.py configs/determinism-stress.yaml
```

## Test provenance sufficiency

After completing a base run:

```bash
python scripts/run_provenance_ablations.py \
  configs/pythia-160m.yaml \
  --run-dir runs/pythia-160m \
  --output runs/pythia-160m/provenance-ablations
```

This intentionally corrupts one execution-plan field at a time and measures divergence from the original run.

## Benchmark checkpoint spacing

First train with checkpoints frequent enough to support the desired sweep. Then:

```bash
python scripts/run_checkpoint_sweep.py \
  configs/pythia-410m-deletion-geometry.yaml \
  --run-dir runs/pythia-410m-deletion-geometry \
  --scenario late-1pct \
  --intervals 250 500 1000 2000 \
  --output results/checkpoint-sweep.json
```

The result reports retained checkpoint bytes and replay latency for each hypothetical checkpoint interval.

## Physically redact a prepared dataset

```bash
python scripts/redact_dataset.py \
  data/prepared/wikitext103-pythia-256 \
  runs/pythia-410m/forget/late-1pct/forget_ids.txt \
  data/redacted/late-1pct
```

## Expand duplicate closure

```bash
python scripts/build_forget_closure.py \
  data/prepared/wikitext103-pythia-256 \
  request_ids.txt \
  closed_request_ids.txt \
  --exact-content \
  --near-hamming 3
```

Near-duplicate closure uses a deterministic 64-bit SimHash over word shingles. The Hamming threshold is an experimental parameter and must be reported with results.

## Approximate unlearning baselines

The repo includes controlled GA, GradDiff, and NPO implementations for same-model comparisons:

```bash
python scripts/run_approximate_baselines.py \
  configs/pythia-410m-deletion-geometry.yaml \
  --state runs/pythia-410m-deletion-geometry/original/final-model-state.pt \
  --forget-ids runs/pythia-410m-deletion-geometry/forget/late-1pct/forget_ids.txt \
  --output runs/pythia-410m-deletion-geometry/approximate-baselines
```

See [`docs/BASELINES.md`](docs/BASELINES.md) for objective definitions and the standardized-benchmark interoperability plan.

## Audit a saved model

```bash
python scripts/audit_saved_model.py \
  configs/pythia-410m-deletion-geometry.yaml \
  --state runs/pythia-410m-deletion-geometry/forget/late-1pct/oracle/final-model-state.pt \
  --forget-ids runs/pythia-410m-deletion-geometry/forget/late-1pct/forget_ids.txt \
  --output results/oracle-audit.json
```

A separately prepared validation token store can be supplied with `--validation-dir` for held-out perplexity and loss-based membership inference.

## Benchmark recent exact rollback

```bash
python scripts/benchmark_rollback.py \
  runs/pythia-410m-deletion-geometry/original/checkpoints \
  --output results/rollback-benchmark.json
```

The benchmark records patch creation time, application time, patch payload bytes, and exact recovery hashes.

## Run cohort-scoped LoRA deletion

```bash
python scripts/run_lora_cohort.py \
  configs/pythia-410m-deletion-geometry.yaml \
  --cohort-ids cohort_ids.txt \
  --output runs/lora-cohort
```

The result reports the frozen base hash before adapter training, the base hash after adapter unload, adapter bytes, and whether base recovery is byte exact.

## Run the approximate curvature hot path

```bash
python scripts/run_hotpath.py \
  configs/pythia-410m-deletion-geometry.yaml \
  --state runs/pythia-410m-deletion-geometry/original/final-model-state.pt \
  --forget-ids runs/pythia-410m-deletion-geometry/forget/late-1pct/forget_ids.txt \
  --output runs/hotpath-late-1pct
```

This method is explicitly approximate. It is not eligible for an exact deletion claim.

## Aggregate results

```bash
python scripts/aggregate_results.py runs \
  --csv results/experiment_rows.csv \
  --json results/experiment_rows.json
```

## Repository map

```text
src/unlearning_at_scale/
  ablations.py      provenance-field ablations
  audit.py          forget, retain, membership, and canary audits
  baselines.py      GA, GradDiff, and NPO baselines
  benchmark.py      rollback benchmark utilities
  compare.py        exact tensor comparison
  dataset.py        immutable and physically redacted token stores
  determinism.py    deterministic controls and environment capture
  duplicates.py     SimHash near-duplicate detection
  experiment.py     end-to-end exact replay experiment
  forget.py         deletion geometry and duplicate closure
  hotpath.py        approximate diagonal-Fisher anti-update
  lora.py           cohort-scoped LoRA support
  losses.py         sum-reduced causal-LM objective
  modeling.py       Hugging Face model loader
  plan.py           immutable microbatch execution plan
  repacked.py       retain-only repacking counterfactual
  results.py        result aggregation
  rollback.py       exact XOR byte patches
  state.py          checkpoints and state hashing
  training.py       deterministic trace runner
  wal.py            fixed-width WAL and manifest reconstruction
```

## Experimental run order

The full sequence is documented in [`docs/RUNBOOK.md`](docs/RUNBOOK.md). In short:

1. freeze code and dependencies
2. pin model revisions
3. prepare and hash datasets
4. run regression tests
5. train original models and record WALs
6. verify no-deletion identity replay
7. run deletion oracles
8. run replay policies with redacted stores
9. run repacked counterfactuals
10. run deletion geometry
11. run model scaling
12. run determinism stress tests
13. run provenance ablations
14. run checkpoint tradeoff sweeps
15. run duplicate closure studies
16. run approximate baselines
17. run rollback, LoRA, and hot-path extensions
18. run audits
19. aggregate and freeze result artifacts
20. populate the claims ledger
21. rewrite the paper

## Scientific claim discipline

[`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md) lists candidate claims and the evidence required before each can appear in the rebuilt manuscript.

No result in the README is a substitute for a released experiment artifact. The current repository contains the experimental machinery. The multi-model GPU study has not yet been executed in this repository.

## Reproducibility

Every release run captures the software stack, Git commit when available, PyTorch/CUDA/cuDNN versions, device information, deterministic settings, model revision, plan hash, WAL hash, checkpoint hashes, and state hashes.

`release_mode: true` rejects a floating model revision such as `main` or `master`.

See:

- [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md)
- [`docs/BASELINES.md`](docs/BASELINES.md)
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)

## Status

**Experimental framework:** implemented

**Local regression suite:** implemented

**Multi-model GPU results:** not yet claimed

**Paper rewrite:** intentionally deferred until the experimental artifact is complete and results are frozen

## License

Apache-2.0.
