# Experimental runbook

The paper is rewritten only after this sequence produces a frozen release bundle.

## Phase A. Freeze code and external frameworks

1. Freeze the project Git commit.
2. Fetch pinned upstreams with `python scripts/bootstrap_upstreams.py`.
3. Verify OpenUnlearning, MUSE and Pythia reference checkouts match `external/upstreams.lock.yaml` exactly.
4. Pin the Python environment and capture package versions.
5. Record GPU, CUDA, cuDNN and PyTorch versions.

## Phase B. Prepare Study A data

Prepare the WikiText-103 trace corpus and validation data using the configured Pythia tokenizer. Preserve the preparation manifest and raw preparation logs. Do not regenerate an artifact under the same name later.

## Phase C. Resolve benchmark sources and prepare Study B data

Resolve the Llama 3.2 1B and TOFU Hub refs. Prepare TOFU with `scripts/prepare_tofu_openunlearning.py`, using the exact resolved model and dataset commits. The exporter must produce `labels.npy` with prompt/system labels masked and only the final assistant response active.

Verify `forget01`, `forget05`, `forget10` and corresponding retain ID files were produced.

## Phase D. Freeze all artifacts

Run:

```bash
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

The generated lock must contain full 40-character Hub commits, exact external Git commits, per-file hashes for both prepared token stores and a directory digest for each dataset. From this point onward, changing any locked data requires a new release identity.

## Phase E. Software verification

```bash
pytest
uas core-smoke --output runs/core-smoke
```

A failure blocks paid runs. The tiny smoke model is never scientific evidence.

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

## Phase G. Study A identity replay

Replay with no deletion. Record model hash equality, optimizer hash equality, unequal elements, maximum absolute difference and replay time. If identity replay fails, deletion replay is not interpreted until the failure is understood.

## Phase H. Study A deletion oracle and replay

For each deletion scenario:

1. select the forget set
2. expand duplicate closure when configured
3. locate the first affected logical step
4. select the latest eligible checkpoint
5. execute the independent trace-preserving deletion oracle
6. materialize a token store with forgotten rows physically absent
7. run `slot_mask`
8. run `filter`
9. compare model and optimizer state with the oracle
10. run the repacked counterfactual

## Phase I. Study A deletion geometry

Use Pythia 410M for early 0.1%, middle 1%, late 1%, random 5%, canary deletion and duplicate closure. Keep geometry separate from the expensive scale axis.

## Phase J. Study A scaling

Run the canonical deletion scenario on 160M, 410M, 1B and 1.4B. Run 2.8B only if budget permits. All model refs are resolved from the same artifact lock.

## Phase K. Determinism envelope

Change one factor at a time: FP32/BF16, eager/SDPA, dropout, strict determinism, optimizer foreach/fused, CPU where feasible and a second GPU architecture if available. Failures remain in the result table.

## Phase L. Provenance sufficiency

Ablate RNG seed, learning rate, sample order, microbatch assignment, accumulation boundary and logical optimizer-step metadata. A field is described as necessary only if evidence supports the statement under the tested regime.

## Phase M. Checkpoint economics

Measure retained checkpoint bytes, nearest eligible checkpoint, replay distance, wall time and exactness across checkpoint policies. Build the storage-versus-recovery Pareto curve from measured data.

## Phase N. Study B replayable TOFU target

Run `configs/benchmarks/tofu-llama32-1b-trace.yaml` through `scripts/run_release.py`. This produces a replayable Llama 3.2 1B target with the benchmark-faithful TOFU labels plus exact deletion outputs for forget01, forget05 and forget10.

The local target is intentionally distinguished from OpenUnlearning's published target checkpoint. The published target is a calibration reference. Direct method comparisons must start from one common target.

## Phase O. Export Study B states to Hugging Face format

For every target/oracle/replay state to be evaluated externally, run:

```bash
python scripts/export_hf_checkpoint.py \
  configs/benchmarks/tofu-llama32-1b-trace.yaml \
  --state <STATE_DICT> \
  --output <HF_CHECKPOINT_DIR>
```

This reconstructs the model on the locked base revision and exports a normal Hugging Face checkpoint directory.

## Phase P. Official TOFU evaluation

Evaluate exported exact replay, oracle and relevant comparison states with the pinned OpenUnlearning evaluator:

```bash
python scripts/openunlearning_adapter.py tofu-eval \
  --checkpoint <HF_CHECKPOINT_DIR> \
  --forget-split forget10
```

Repeat for forget01, forget05 and forget10. The adapter records the upstream commit and evaluation provenance.

## Phase Q. Official approximate baselines

Run GradAscent, GradDiff, NPO and SimNPO through the pinned OpenUnlearning framework. For any direct comparison to our replay method, pass the same local target checkpoint and the same retain reference used in evaluation.

Our `src/unlearning_at_scale/baselines.py` implementations are sanity/debug implementations only and do not supply publication baseline rows.

## Phase R. Operational extensions

Measure XOR rollback patch size/latency/exact recovery, cohort LoRA base recovery and the approximate curvature hot path. Keep these secondary to the exact replay result.

## Phase S. Study C MUSE, budget permitting

Only after Studies A and B are complete, run the pinned OpenUnlearning MUSE integration on News and Books. Include the base six-way evaluation, deletion-size scaling and sequential-deletion sustainability where budget permits. Record both the OpenUnlearning commit and original MUSE commit.

## Phase T. Audits and aggregation

Run state equality checks plus relevant behavioral audits. Aggregate all rows with:

```bash
python scripts/aggregate_results.py runs \
  --csv results/experiment_rows.csv \
  --json results/experiment_rows.json
```

Do not manually copy values into paper tables.

## Phase U. Release bundle

Archive resolved configs, `locks/artifacts.lock.json`, environment snapshots, dataset manifests, execution plans, WALs, ordered-ID manifests, checkpoint hashes, final state hashes, forget-ID lists, OpenUnlearning command manifests, audit JSON and aggregated tables. Large model weights may live outside GitHub but their immutable location and hash must be recorded.

## Phase V. Claims ledger

Populate every evidence cell in `docs/CLAIMS_LEDGER.md`. Delete unsupported claims.

## Phase W. Paper rewrite

Only after the artifact is frozen:

1. rewrite the abstract from actual results
2. rewrite introduction and contributions
3. write the literature review
4. formalize the trace-preserving counterfactual
5. describe the released system
6. write methodology from frozen configs
7. generate tables/figures from machine-readable results
8. write failure modes and limitations from observed evidence
9. keep the paper title unchanged
