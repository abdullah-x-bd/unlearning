# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models

[![CI](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml)

Experimental artifact for a ground-up reconstruction of **Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models**.

The central question is not only how to edit a trained model after a deletion request. It is whether training itself can be engineered so that a future deletion request becomes a reproducible counterfactual computation.

## Core idea

Most machine-unlearning methods start from a trained model and try to remove the influence of a forget set afterward.

This project instead records enough of the original training execution to define and reproduce a **trace-preserving deletion counterfactual**.

The execution trace fixes:

- sample slots
- microbatch seeds
- optimizer-step boundaries
- learning-rate values
- accumulation structure

When records are forgotten, their original slots remain in the trace but contribute zero loss. The records themselves can then be physically removed from the replay data store.

This distinction matters because ordinary retain-only retraining changes the computation. Repacking retained examples can change batch shapes, RNG consumption, dropout masks, accumulation boundaries, optimizer steps, learning-rate schedules, and floating-point reduction order.

The exact target in this repository is therefore:

```text
original training trace
        |
        +--> full-data replay
        |
        +--> same trace with forgotten slots contributing zero loss
                         |
                         +--> deletion oracle
                         |
                         +--> replay after physical deletion
```

The final test is state equality, not merely similar behavior.

## Frozen results

Two paid GPU releases are complete and frozen.

| Release | Hardware | Identity replay | Deletion replay | Physical redaction |
| --- | --- | --- | --- | --- |
| Pythia 160M | RTX A6000 | exact | exact for early 0.1%, middle 1%, late 1%, random 5% | yes |
| Pythia 2.8B | A100-SXM4-80GB | exact | exact for random 5% | yes |

### Pythia 160M

Identity replay reproduced the original model and optimizer exactly across **162,322,944 model elements**.

For all four tested deletion geometries, slot-preserving replay matched the trace-preserving deletion oracle exactly. Physical filtering and densely repacked retain-only retraining diverged from the oracle in every tested scenario.

Release record:
[`results/releases/pythia-160m-2026-08-11/`](results/releases/pythia-160m-2026-08-11/README.md)

### Pythia 2.8B

Identity replay reproduced the original run exactly across **388 tensors and 2,775,208,960 model elements**:

- unequal tensors: `0`
- unequal elements: `0`
- maximum absolute difference: `0.0`
- L2 difference: `0.0`
- optimizer hash equal: `true`

For a random 5% deletion request, **1,013 of 20,256 records** were physically removed. Replay from the redacted store reproduced the deletion oracle exactly across all 2,775,208,960 model elements and the optimizer state.

Release record:
[`results/releases/pythia-2.8b-2026-08-11/`](results/releases/pythia-2.8b-2026-08-11/README.md)

## What the results establish

Under the pinned single-GPU environments used in the frozen releases, the evidence supports the following systems claims:

- deterministic no-deletion replay can reproduce the original model and optimizer exactly
- trace-preserving deletion replay can reproduce the deletion counterfactual exactly
- exact replay remains possible after forgotten token rows have been physically removed from the replay store
- preserving logical slots is materially different from physically filtering or densely repacking retained examples
- the result scales from Pythia 160M to Pythia 2.8B in the tested environments

The evidence does **not** establish:

- multi-GPU exactness
- exactness under arbitrary hardware or software stacks
- standardized semantic-unlearning performance on TOFU or MUSE
- superiority over approximate methods such as NPO or GradDiff
- direct satisfaction of any legal erasure obligation

The current claim boundary is maintained in [`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md).

## Provenance and replay system

The implementation in `src/unlearning_at_scale/` includes:

- immutable execution plans
- deterministic per-microbatch RNG seeding
- fixed gradient-accumulation boundaries
- float32 learning-rate values in the trace
- checkpoint recovery
- exact model and optimizer hashing
- trace-preserving deletion oracles
- slot-preserving redacted replay
- physical filtering and repacked counterfactuals
- physical token-store redaction
- provenance-field ablations
- checkpoint-spacing sweeps
- XOR rollback experiments
- cohort-scoped LoRA experiments
- approximate curvature-based deletion experiments

### WAL format

Each binary WAL record is exactly **32 bytes**:

```text
uint64 ordered_sample_ids_digest
uint64 microbatch_seed
float32 learning_rate
uint32 logical_optimizer_step
uint16 microbatch_length
uint16 flags
uint32 crc32
```

Ordered sample IDs are stored separately in a manifest. The repository reports WAL bytes, manifest bytes, and total provenance cost independently.

For the frozen releases:

| Release | WAL records | WAL bytes | Ordered-ID manifest | Total provenance |
| --- | ---: | ---: | ---: | ---: |
| Pythia 160M | 5,064 | 162,048 | 1,134,250 | 1,296,298 |
| Pythia 2.8B | 20,256 | 648,192 | 3,453,690 | 4,101,882 |

The 32-byte figure applies only to the fixed binary WAL record. It is not the total provenance footprint.

## Experimental program

### Study A: controlled systems and scaling

Pythia + WikiText-103 isolates systems questions such as exactness, deletion geometry, replay policy, checkpoint placement, provenance cost, physical redaction, and model scale.

Completed publication-scale releases:

- Pythia 160M
- Pythia 2.8B

Intermediate Pythia sizes remain available but are no longer prerequisites for the central scaling claim.

### Study B: standardized TOFU validation

The next empirical priority is **Llama 3.2 1B Instruct + TOFU** using the pinned OpenUnlearning framework.

The repository already contains:

- OpenUnlearning-compatible TOFU preprocessing
- answer-only label masks matching the benchmark supervision structure
- `forget01`, `forget05`, and `forget10` split support
- checkpoint export for external evaluation
- a pinned OpenUnlearning adapter
- publication-authoritative GradAscent, GradDiff, NPO, and SimNPO hooks

The purpose of Study B is different from Study A. Study A tests exact state reconstruction. Study B tests whether the resulting exact deletion counterfactual behaves appropriately under standardized unlearning metrics and how it compares with established approximate methods.

See [`docs/EXTERNAL_BENCHMARKS.md`](docs/EXTERNAL_BENCHMARKS.md) and [`docs/BASELINES.md`](docs/BASELINES.md).

### Study C: MUSE

MUSE remains an optional external-validation extension after TOFU. It is not required before the main manuscript is rebuilt.

## Reproducibility

Remote models, datasets, and research repositories are pinned before publication runs.

`locks/artifact-sources.yaml` defines source artifacts. `locks/artifacts.lock.json` records resolved Hugging Face commit SHAs, external Git commits, prepared-dataset file hashes, and directory digests.

Publication runners refuse to execute when the frozen lock no longer verifies.

Pinned external research code includes:

- OpenUnlearning at `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- MUSE reference implementation at `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e`
- Pythia reference repository at `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`

Each frozen release records the resolved model revision, execution-plan hash, WAL and manifest hashes, environment snapshot, GPU probe, result summaries, workflow identifiers, artifact digest, and cost metadata.

## Quick start

Install the project and run the software test suite:

```bash
python -m pip install -e '.[dev]'
pytest
uas core-smoke --output runs/core-smoke
```

Prepare and verify frozen research dependencies:

```bash
python scripts/bootstrap_upstreams.py
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

Run a locked release configuration:

```bash
python scripts/run_release.py configs/pythia-160m.yaml
```

The full experimental sequence is documented in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Repository map

```text
src/unlearning_at_scale/   core replay and deletion system
configs/                   experiment configurations
scripts/                   data, release, benchmark, and audit tooling
locks/                     frozen artifact and environment metadata
external/                  pinned upstream definitions
results/releases/          immutable empirical release records
docs/                      experiment design, claims, baselines, reproducibility
```

## Research status

**Core replay system:** implemented

**Frozen systems evidence:** Pythia 160M and Pythia 2.8B complete

**Exact 2.8B result:** complete

**Standardized TOFU/OpenUnlearning validation:** next empirical priority

**MUSE:** optional after TOFU

**Paper rewrite:** downstream of the final evidence set

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0.
