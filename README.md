# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models

[![CI](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml)

This repository is the experimental artifact for a ground-up reconstruction of **Unlearning at Scale**. The research question is whether LLM training can be engineered so that a later data-deletion request becomes a reproducible counterfactual computation rather than an ad hoc post-training edit.

The repository deliberately separates what is implemented from what has been empirically established. The small model used in CI is a regression test only. Paper claims will be drawn from the multi-model GPU experiments and released result artifacts.

## Core idea

A training run is represented as a program with a fixed execution plan. For each microbatch, the system records the ordered sample-ID digest, a microbatch RNG seed, the exact float32 learning-rate value, the logical optimizer-step counter, the microbatch length, and the accumulation boundary. The binary WAL record is exactly **32 bytes**. The ordered-ID manifest is separate and is measured explicitly, so the repository does not treat 32 bytes as the total provenance cost.

For a future forget set `F`, replay begins from the latest checkpoint that predates the first affected logical step. The forgotten records then contribute zero training loss while the original execution plan remains fixed.

The exact target in this repository is therefore a **trace-preserving deletion counterfactual**, not an unspecified fresh training run on a repacked `D \ F` dataset.

That distinction matters. Repacking can alter microbatch composition, optimizer-step boundaries, random-number consumption, floating-point reduction order, and the learning-rate trajectory. The repo measures repacked retraining separately instead of silently assuming equivalence.

## What is being tested

The main study asks five questions.

1. Is a compact WAL sufficient to reconstruct the training execution plan from a clean checkpoint?
2. Under which real training configurations does replay match an independent deletion oracle byte for byte?
3. Does slot-preserving masked replay improve exactness over physically filtering rows from a microbatch?
4. Where does determinism fail as model size, dtype, attention kernels, dropout, optimizer implementation, and hardware change?
5. What are the storage and latency tradeoffs of checkpoints, replay, recent XOR rollback, and cohort-scoped adapters?

Failures are results. The experiment is not designed to force every configuration to pass.

## Main model matrix

The initial scaling suite uses EleutherAI Pythia models:

- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b` as the budget-dependent extension

Pythia is useful here because it provides a consistent research-oriented model family across parameter scales. Release runs will pin immutable model revisions rather than `main`.

## Data

The default data-preparation script uses WikiText-103 raw text, converts it into fixed-length causal-LM records, assigns immutable sample IDs and content hashes, and injects synthetic canary groups for controlled memorization and extraction audits.

The prepared token arrays are not committed to Git.

Prepare the dataset with:

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

## Replay semantics

### `slot_mask`

The original microbatch shape is retained. A forgotten sample ID is replaced by a deterministic dummy sequence and its token-loss weight is set to zero. The forgotten content is therefore unnecessary during replay, while the batch slot and RNG position remain stable.

### `filter`

Forgotten rows are physically removed from the microbatch before the forward pass. This is mathematically clean under a sum-reduced loss but can alter batch shapes and numerical execution. It is tested rather than assumed exact.

### `repacked`

Retained records are densely repacked into new microbatches and a normal schedule is rebuilt. This is a diagnostic alternative counterfactual, not the exact target used by the replay proof.

## Fixed-width WAL

Each WAL record uses the binary layout:

```text
uint64 ordered_sample_ids_digest
uint64 microbatch_seed
float32 learning_rate
uint32 logical_optimizer_step
uint16 microbatch_length
uint16 flags
uint32 crc32
```

Total binary WAL record: **32 bytes per microbatch**. This excludes the ordered-ID manifest. Every run reports WAL bytes, manifest bytes, and their combined provenance cost.

The ordered IDs themselves live in an access-controlled manifest. Public research runs use SHA-256-derived digests. Production use should set `UNLEARNING_WAL_HMAC_KEY`, which switches the digest to HMAC-SHA256 before truncation.

The WAL file receives a segment-level SHA-256 in addition to per-record CRC32 checks.

## Exactness evidence

Every oracle/replay comparison records:

- full model-state SHA-256
- optimizer-state SHA-256
- unequal tensor count
- unequal element count
- maximum absolute parameter difference
- L2 parameter difference
- applied optimizer updates
- skipped logical updates
- replay wall time
- WAL size
- checkpoint size

A byte-exact claim is made only when the hashes and tensor comparison agree.

## Experiment matrix

The default forget scenarios are:

| Scenario | Position | Fraction |
| --- | --- | ---: |
| `early-0.1pct` | early training | 0.1% |
| `middle-1pct` | middle training | 1% |
| `late-1pct` | late training | 1% |
| `random-5pct` | distributed | 5% |

The determinism stress suite additionally varies FP32/BF16, eager/SDPA attention, dropout policy, deterministic-algorithm enforcement, optimizer implementation, CPU/GPU execution, microbatch size, accumulation length, and checkpoint distance.

See [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Run the dependency-light regression test

```bash
python -m pip install -e '.[dev]'
uas core-smoke --output runs/core-smoke
```

This validates WAL serialization, WAL-to-plan reconstruction, checkpoint restoration, and exact trace replay without downloading a model. It is software testing only and is never used as evidence in the paper.

## Run a Pythia experiment

After preparing the token dataset:

```bash
uas run configs/pythia-160m.yaml
```

Run the scale study with:

```bash
python scripts/run_matrix.py configs/matrix-main.yaml
```

The main scale matrix uses one canonical late-deletion scenario across 160M, 410M, 1B, 1.4B, and 2.8B. Deletion geometry is then studied separately on 410M with `configs/pythia-410m-deletion-geometry.yaml`, and one-factor-at-a-time determinism stress tests run through `scripts/run_stress.py`. This staged design avoids spending large-model GPU budget on a needlessly Cartesian experiment grid.

## Repository map

```text
src/unlearning_at_scale/
  audit.py          secondary forget/retain audits
  compare.py        exact tensor comparison
  dataset.py        immutable token store and redacted replay batches
  determinism.py    deterministic execution controls and environment capture
  experiment.py     end-to-end experiment runner
  forget.py         forget-set selection and duplicate closure
  hotpath.py        approximate diagonal-Fisher anti-update
  lora.py           cohort-scoped LoRA support
  losses.py         sum-reduced causal-LM objective
  modeling.py       Hugging Face causal-LM loader
  plan.py           immutable microbatch execution plan
  repacked.py       ordinary retain-set repacking baseline
  rollback.py       exact XOR tensor rollback patches
  state.py          checkpoints and canonical state hashing
  training.py       deterministic trace runner
  wal.py            32-byte WAL and manifest reconstruction
```

## Operational extensions

The rebuilt paper will keep the exact replay path central. Three operational extensions remain in the artifact so their claims can be tested rather than asserted.

**Recent rollback.** `rollback.py` stores XOR byte patches for tensors plus prior scalar metadata. Applying the patch to the later state reconstructs the earlier bytes exactly when the state structure is unchanged.

**Cohort adapters.** `lora.py` supports LoRA adapters over a frozen base. A cohort adapter can be removed without changing the base model. Scaled experiments still need to quantify utility, storage, and operational latency.

**Approximate hot path.** `hotpath.py` contains a diagonal-Fisher curvature anti-update. It is explicitly approximate. It is evaluated with forget, retain, extraction, and membership audits and should be viewed only as a temporary path before exact replay.

## Scientific claim discipline

The old manuscript is not treated as evidence. The new paper will be rewritten from the released experimental results.

[`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md) lists each candidate claim and the evidence required before it may enter the manuscript. In particular, no distributed exactness claim is permitted until a dedicated multi-GPU experiment exists.

## Reproducibility

PyTorch reproducibility is scoped to a specific software and hardware environment. Every release run must capture the exact Python packages, PyTorch/CUDA/cuDNN versions, device model, deterministic settings, model revision, dataset hashes, plan hash, WAL hash, and checkpoint hashes.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md).

## Status

**Repository implementation:** active reconstruction

**Core WAL/replay/rollback regression tests:** implemented

**Multi-model GPU results:** not yet claimed

**Rewritten paper:** deliberately deferred until the GitHub evidence is complete

## License

Apache-2.0.
