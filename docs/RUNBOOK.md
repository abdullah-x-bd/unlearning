# Experimental runbook

This document records the intended publication workflow and the current frozen campaign status. The paper is rewritten only from released evidence and machine-readable outputs.

## Current frozen campaign status

Completed publication-facing evidence:

- Pythia 160M exactness release with four deletion geometries
- Pythia 2.8B exactness scale release with random 5% deletion
- Llama 3.2 1B Instruct + TOFU `forget10` exactness release
- physical token-store redaction for every frozen deletion release
- hash-verified reconstruction of the frozen Llama original and deletion states
- standardized OpenUnlearning TOFU evaluation of those frozen states
- durable release records under `results/releases/`

Not part of the current frozen evidence set:

- matched local retain-only TOFU control
- publication-authoritative GradAscent, GradDiff, NPO and SimNPO comparison runs
- provenance-field ablations at publication scale
- checkpoint-spacing Pareto sweep at publication scale
- multi-GPU deterministic exactness
- MUSE extension

The original planning matrix below contains broader optional phases than were ultimately required for the first frozen release. Where the executed campaign differed from the initial plan, the completed release records and `docs/CLAIMS_LEDGER.md` are authoritative.

## Phase A. Freeze code and external frameworks

1. Freeze the project Git commit.
2. Fetch pinned upstreams with `python scripts/bootstrap_upstreams.py`.
3. Verify OpenUnlearning, MUSE and Pythia reference checkouts match `external/upstreams.lock.yaml` exactly.
4. Pin the Python environment and capture package versions.
5. Record GPU, CUDA, cuDNN and PyTorch versions.

**Status:** complete for the frozen releases.

## Phase B. Prepare Study A data

Prepare the WikiText-103 trace corpus and validation data using the configured Pythia tokenizer. Preserve the preparation manifest and raw preparation logs. Do not regenerate an artifact under the same name later.

**Status:** complete for the frozen Pythia releases.

## Phase C. Resolve benchmark sources and prepare Study B data

Resolve the Llama 3.2 1B and TOFU Hub refs. Prepare TOFU with `scripts/prepare_tofu_openunlearning.py`, using the exact resolved model and dataset commits. The exporter must produce `labels.npy` with prompt/system labels masked and only the final assistant response active.

Verify `forget01`, `forget05`, `forget10` and corresponding retain ID files were produced.

**Status:** preparation complete; the current frozen benchmark-facing release uses `forget10`.

## Phase D. Freeze all artifacts

Run:

```bash
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

The generated lock must contain full 40-character Hub commits, exact external Git commits, per-file hashes for both prepared token stores and a directory digest for each dataset. From this point onward, changing any locked data requires a new release identity.

**Status:** complete for the frozen campaign.

## Phase E. Software verification

```bash
pytest
uas core-smoke --output runs/core-smoke
```

A failure blocks paid runs. The tiny smoke model is never scientific evidence.

**Status:** release gates passed before the frozen paid runs.

## Phase F. Study A original training and provenance

Use `scripts/run_release.py` or `scripts/run_release_matrix.py`, never the unlocked runner, for publication evidence.

For each Pythia model:

1. load the full SHA from the artifact lock
2. save the step-zero checkpoint
3. build the immutable execution plan
4. train the original trace
5. record WAL and ordered-ID manifest
6. save periodic checkpoints
7. save final model and optimizer hashes
8. record runtime and storage

**Status:** complete at the frozen 160M and 2.8B endpoints.

## Phase G. Study A identity replay

Replay with no deletion. Record model hash equality, optimizer hash equality, unequal elements, maximum absolute difference and replay time. If identity replay fails, deletion replay is not interpreted until the failure is understood.

**Status:** exact at Pythia 160M and Pythia 2.8B.

## Phase H. Study A deletion oracle and replay

For each deletion scenario:

1. select the forget set
2. expand duplicate closure when configured
3. locate the first affected logical step
4. select the latest eligible checkpoint
5. execute the independent trace-preserving deletion oracle
6. materialize a token store with forgotten rows physically absent
7. run `slot_mask`
8. run `filter` where that comparison is part of the study
9. compare model and optimizer state with the oracle
10. run the repacked counterfactual where that comparison is part of the study

**Status:** exact slot-preserving deletion replay is frozen at Pythia 160M across four scenarios and at Pythia 2.8B for random 5%. Filter/repacked controls were intentionally concentrated in the Pythia 160M systems study rather than repeated at every scale.

## Phase I. Study A deletion geometry

The original plan proposed a broader intermediate-scale geometry matrix. The executed frozen campaign instead concentrates deletion-geometry evidence at Pythia 160M, where early 0.1%, middle 1%, late 1%, and random 5% scenarios are all frozen.

**Status:** complete for the four frozen 160M scenarios; additional geometry remains optional.

## Phase J. Study A scaling

The original plan considered a dense sequence of intermediate Pythia sizes. The publication-facing campaign uses Pythia 160M as the controlled systems endpoint and Pythia 2.8B as the direct scale check.

**Status:** complete for the frozen 160M-to-2.8B scale comparison.

## Phase K. Determinism envelope

Change one factor at a time: FP32/BF16, eager/SDPA, dropout, strict determinism, optimizer foreach/fused, CPU where feasible and a second GPU architecture if available. Failures remain in the result table.

**Status:** implementation and selected environment checks exist; a full publication-scale determinism envelope is not part of the current frozen claim set.

## Phase L. Provenance sufficiency

Ablate RNG seed, learning rate, sample order, microbatch assignment, accumulation boundary and logical optimizer-step metadata. A field is described as necessary only if evidence supports the statement under the tested regime.

**Status:** implementation complete; publication-scale runs pending.

## Phase M. Checkpoint economics

Measure retained checkpoint bytes, nearest eligible checkpoint, replay distance, wall time and exactness across checkpoint policies. Build the storage-versus-recovery Pareto curve from measured data.

**Status:** implementation complete; full publication-scale sweep pending.

## Phase N. Study B replayable TOFU target

Run the locked Llama 3.2 1B + TOFU configuration through the publication release path. The frozen campaign trains the 4,000-example full target and evaluates exact trace-preserving deletion for `forget10`.

The local target is intentionally distinguished from OpenUnlearning's published target checkpoint. The published target/retain checkpoints are calibration references. Direct method comparisons must start from one common target.

**Status:** complete for Llama 3.2 1B + TOFU `forget10`.

## Phase O. Export Study B states to Hugging Face format

For every target/oracle/replay state to be evaluated externally, run:

```bash
python scripts/export_hf_checkpoint.py \
  configs/benchmarks/tofu-llama32-1b-trace.yaml \
  --state <STATE_DICT> \
  --output <HF_CHECKPOINT_DIR>
```

This reconstructs the model on the locked base revision and exports a normal Hugging Face checkpoint directory.

**Status:** complete for the hash-verified frozen original and `forget10` deletion states used in OpenUnlearning evaluation.

## Phase P. Official TOFU evaluation

Evaluate exported frozen states with the pinned OpenUnlearning evaluator:

```bash
python scripts/openunlearning_adapter.py tofu-eval \
  --checkpoint <HF_CHECKPOINT_DIR> \
  --forget-split forget10
```

The adapter records the upstream commit and evaluation provenance. The recovery workflow additionally verifies the canonical state hashes before evaluation, preserves the exact evaluator logs, and fails closed unless the result artifact is locally revalidated and uploaded before Pod termination.

The OpenUnlearning path applies a recorded BF16-to-float32 NumPy interoperability cast immediately before NumPy conversion. The patch is frozen with original/patched hashes and exact replacements and does not change the BF16 metric values or evaluator logic.

**Status:** complete for the frozen original and trace-preserving `forget10` deletion states. Durable record: `results/releases/tofu-openunlearning-eval-2026-08-12/`.

## Phase Q. Official approximate baselines

Run GradAscent, GradDiff, NPO and SimNPO through the pinned OpenUnlearning framework. For any direct comparison to our replay method, pass the same local target checkpoint and the same retain reference used in evaluation.

Our `src/unlearning_at_scale/baselines.py` implementations are sanity/debug implementations only and do not supply publication baseline rows.

**Status:** integration complete; publication-authoritative runs pending.

A matched local retain-only control should also be run before making a causal claim that the behavioral difference from the official `retain90` reference is caused by trace preservation versus dense repacking.

## Phase R. Operational extensions

Measure XOR rollback patch size/latency/exact recovery, cohort LoRA base recovery and the approximate curvature hot path. Keep these secondary to the exact replay result.

**Status:** implementation exists; scale evidence remains optional/pending unless promoted into the manuscript.

## Phase S. Study C MUSE, budget permitting

Only after the core Studies A and B evidence is frozen, run the pinned OpenUnlearning MUSE integration on News and Books if additional external validation is warranted. Include the base six-way evaluation, deletion-size scaling and sequential-deletion sustainability where budget permits. Record both the OpenUnlearning commit and original MUSE commit.

**Status:** optional, not required for the current release.

## Phase T. Audits and aggregation

Run state equality checks plus relevant behavioral audits. Aggregate rows from machine-readable outputs where applicable. Do not manually copy values into paper tables when a machine-readable source can be used.

**Status:** state-equality releases and OpenUnlearning behavioral evaluation are frozen. Any new matched-control or approximate-baseline rows must follow the same evidence policy.

## Phase U. Release bundle

Archive resolved configs, `locks/artifacts.lock.json`, environment snapshots, dataset manifests, execution plans, WALs, ordered-ID manifests, checkpoint hashes, final state hashes, forget-ID lists, OpenUnlearning command manifests, audit JSON and aggregated tables. Large model weights may live outside GitHub but their immutable location and hash must be recorded.

For the first GitHub repository release, attach the compact frozen evidence bundles in addition to the source snapshot so the scientific record does not depend only on GitHub Actions artifact retention.

## Phase V. Claims ledger

Populate every evidence cell in `docs/CLAIMS_LEDGER.md`. Unsupported claims remain explicitly pending or prohibited.

**Status:** updated through the completed OpenUnlearning evaluation.

## Phase W. Paper rewrite

Only after the repository snapshot is frozen:

1. rewrite the abstract from actual results
2. rewrite introduction and contributions
3. update the literature review
4. formalize the trace-preserving counterfactual
5. describe the released system
6. write methodology from frozen configs
7. generate tables/figures from machine-readable results
8. write failure modes and limitations from observed evidence
9. incorporate standardized behavioral evaluation without conflating it with state exactness
10. preserve the explicit claim boundary around matched retain-only controls, approximate baselines, distributed exactness, privacy, and legal compliance

**Status:** next stage after the first repository release snapshot.
