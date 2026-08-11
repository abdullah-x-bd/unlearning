# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models

[![CI](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml)

Experimental artifact for a ground-up reconstruction of **Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models**.

The central question is whether language-model training can be engineered so that a later data-deletion request becomes a reproducible counterfactual computation rather than an ad hoc post-training edit.

## Core idea

Most machine-unlearning methods start from a trained model and attempt to remove the influence of a forget set afterward. This project instead records enough of the original training execution to define and reproduce a **trace-preserving deletion counterfactual**.

The execution trace fixes sample slots, microbatch seeds, optimizer-step boundaries, learning-rate values, and accumulation structure. When records are forgotten, their original logical slots remain in the trace but contribute zero loss. The records themselves can then be physically removed from the replay data store.

This distinction matters because ordinary retain-only retraining changes the computation. Repacking retained examples can change batch shapes, RNG consumption, dropout masks, accumulation boundaries, optimizer steps, learning-rate schedules, and floating-point reduction order.

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

The primary systems test is state equality, not merely similar behavior.

## Frozen results

Three paid single-GPU releases are complete and frozen.

| Release | Hardware | Identity replay | Deletion replay | Physical redaction |
| --- | --- | --- | --- | --- |
| Pythia 160M | RTX A6000 | exact | exact for early 0.1%, middle 1%, late 1%, random 5% | yes |
| Pythia 2.8B | A100-SXM4-80GB | exact | exact for random 5% | yes |
| Llama 3.2 1B + TOFU | RTX A6000 | exact | exact for `forget10` | yes |

### Pythia 160M

Identity replay reproduced the original model and optimizer exactly across **162,322,944 model elements**. For all four tested deletion geometries, slot-preserving replay matched the trace-preserving deletion oracle exactly. Physical filtering and densely repacked retain-only retraining diverged from the oracle in every tested scenario.

Release record: [`results/releases/pythia-160m-2026-08-11/`](results/releases/pythia-160m-2026-08-11/README.md)

### Pythia 2.8B

Identity replay reproduced the original run exactly across **388 tensors and 2,775,208,960 model elements**. For a random 5% deletion request, **1,013 of 20,256 records** were physically removed. Replay from the redacted store reproduced the deletion oracle exactly across every model element and the optimizer state.

Release record: [`results/releases/pythia-2.8b-2026-08-11/`](results/releases/pythia-2.8b-2026-08-11/README.md)

### Llama 3.2 1B + TOFU forget10

The cross-family benchmark-facing release trained Llama 3.2 1B Instruct on the 4,000-example TOFU full set for five epochs using answer-only supervision.

Identity replay was exact across **147 tensors and 1,498,482,688 model elements**, including optimizer state. The TOFU `forget10` request selected **400 of 4,000 records**. Those rows were physically removed, leaving 3,600 retained rows with the benchmark label masks preserved. Trace-preserving slot replay from that redacted store reproduced the deletion oracle exactly across all 1,498,482,688 model elements and the optimizer state.

Canonical hashes:

- full-data target: `54c711e9bde77215d9c5def50429f925a382bdcd28150bb87a89a118dd54bc65`
- forget10 deletion state: `067109bfd2e34f1616a8069d04ecd28b4814513332b03957ab917503122aeec3`

Release record: [`results/releases/tofu-llama32-1b-forget10-2026-08-11/`](results/releases/tofu-llama32-1b-forget10-2026-08-11/README.md)

The next step is deliberately separate: reconstruct these canonical states under the same pinned RTX A6000 environment, require the frozen hashes to match, and then evaluate the exported checkpoints with the pinned OpenUnlearning TOFU evaluator against the immutable `retain90` reference.

## What the results establish

Under the pinned single-GPU environments used in the frozen releases, the evidence supports the following systems claims:

- deterministic no-deletion replay can reproduce the original model and optimizer exactly
- trace-preserving deletion replay can reproduce the specified deletion counterfactual exactly
- exact replay remains possible after forgotten token rows have been physically removed from the replay store
- preserving logical slots is materially different from physically filtering or densely repacking retained examples in the tested Pythia setting
- the result scales from Pythia 160M to Pythia 2.8B
- state-level exactness also reproduces on a second model family, Llama 3.2 1B, using the standardized TOFU data format

The evidence does **not** yet establish:

- standardized OpenUnlearning forget quality or model utility for the exact deletion state
- superiority over NPO, GradDiff, or other approximate unlearning methods
- multi-GPU exactness
- exactness under arbitrary hardware or software stacks
- direct satisfaction of any legal erasure obligation

The claim boundary is maintained in [`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md).

## Provenance and replay system

The implementation in `src/unlearning_at_scale/` includes immutable execution plans, deterministic per-microbatch RNG seeding, fixed gradient-accumulation boundaries, float32 learning-rate values in the trace, checkpoint recovery, exact model and optimizer hashing, trace-preserving deletion oracles, slot-preserving redacted replay, physical token-store redaction, provenance-field ablations, checkpoint-spacing sweeps, XOR rollback experiments, cohort-scoped LoRA experiments, and approximate curvature-based deletion experiments.

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

| Release | WAL records | WAL bytes | Ordered-ID manifest | Total provenance |
| --- | ---: | ---: | ---: | ---: |
| Pythia 160M | 5,064 | 162,048 | 1,134,250 | 1,296,298 |
| Pythia 2.8B | 20,256 | 648,192 | 3,453,690 | 4,101,882 |
| Llama 3.2 1B + TOFU | 2,500 | 80,000 | 878,890 | 958,890 |

The 32-byte figure applies only to the fixed binary WAL record. It is not the total provenance footprint.

## Experimental program

### Study A: controlled systems and scaling

Pythia + WikiText-103 isolates exactness, deletion geometry, replay policy, checkpoint placement, provenance cost, physical redaction, and model scale. The completed publication-scale endpoints are Pythia 160M and Pythia 2.8B. Intermediate Pythia sizes remain available but are no longer prerequisites for the central scaling result.

### Study B: TOFU and OpenUnlearning

The state-level phase is complete for **Llama 3.2 1B Instruct + TOFU forget10**. The repository now contains:

- OpenUnlearning-compatible TOFU preprocessing
- answer-only label masks matching benchmark supervision
- a frozen exact `forget10` replay release
- canonical original and deletion hashes
- two-pass canonical-state reconstruction for external evaluation
- Hugging Face checkpoint export
- a pinned OpenUnlearning adapter
- an immutable `retain90` reference from the artifact lock
- explicit evaluator attention-backend provenance
- publication-authoritative GradAscent, GradDiff, NPO, and SimNPO hooks for later comparison

The OpenUnlearning evaluation is intentionally downstream of the exactness result. It asks how a known exact trace-preserving deletion counterfactual scores under standardized behavioral unlearning metrics. Approximate baselines are not run until that result is inspected.

See [`docs/EXTERNAL_BENCHMARKS.md`](docs/EXTERNAL_BENCHMARKS.md) and [`docs/BASELINES.md`](docs/BASELINES.md).

### Study C: MUSE

MUSE remains an optional external-validation extension after TOFU. It is not required before the main manuscript is rebuilt.

## Reproducibility

Remote models, datasets, and research repositories are pinned before publication runs. `locks/artifact-sources.yaml` defines source artifacts. `locks/artifacts.lock.json` records resolved Hugging Face commit SHAs, external Git commits, prepared-dataset file hashes, and directory digests. Publication runners refuse to execute when the frozen lock no longer verifies.

Pinned external research code includes:

- OpenUnlearning at `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- MUSE reference implementation at `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e`
- Pythia reference repository at `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`

The OpenUnlearning interoperability path also pins the official Llama 3.2 1B `retain90` checkpoint to commit `7114300c0049527a71833f5683965c358ad9dcbf`. Evaluation uses the same explicit attention backend for the retain reference and both reconstructed project checkpoints, and records the exact commands in `uas_interop.json`.

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

Run a locked systems release configuration:

```bash
python scripts/run_release.py configs/pythia-160m.yaml
```

Dry-run the pinned OpenUnlearning interoperability command without evaluating a model:

```bash
python scripts/openunlearning_adapter.py tofu-eval \
  --checkpoint /tmp/placeholder \
  --forget-split forget10 \
  --attention-implementation eager \
  --dry-run
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

**Frozen exactness evidence:** Pythia 160M, Pythia 2.8B, and Llama 3.2 1B + TOFU forget10 complete

**Cross-family exactness:** complete in the tested single-GPU environments

**Standardized OpenUnlearning behavioral evaluation:** prepared and pending guarded execution

**Approximate OpenUnlearning baselines:** deferred until the exact deletion evaluation is inspected

**MUSE:** optional after TOFU

**Paper rewrite:** downstream of the final evidence set

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE).
