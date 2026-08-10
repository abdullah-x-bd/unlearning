# Experimental runbook

This file defines the order in which the research artifact should be executed. The paper should not be rewritten before the release bundle reaches the end of this runbook.

## Phase A. Freeze the experiment definition

1. Freeze the Git commit used for experiments.
2. Pin Python package versions in the execution environment.
3. Replace floating model revisions with immutable Hugging Face commit SHAs.
4. Set `release_mode: true` in release configs.
5. Record the exact GPU type and CUDA stack.
6. Decide the final model matrix before beginning expensive runs.

## Phase B. Freeze datasets

1. Prepare the WikiText trace dataset.
2. Prepare validation data with the same tokenizer and sequence length.
3. Prepare the TOFU track.
4. Verify all dataset manifests and hashes.
5. Keep raw preparation logs.
6. Do not regenerate a dataset midway through the experiment under the same artifact name.

## Phase C. Software verification

Run:

```bash
pytest
uas core-smoke --output runs/core-smoke
```

The release commit should have a clean regression run before GPU experiments begin.

## Phase D. Original training and provenance

For each primary model:

1. load the pinned base model
2. save the step-zero checkpoint
3. build the immutable execution plan
4. train the original run
5. record the WAL and ordered-ID manifest
6. save periodic checkpoints
7. save final model-state hashes and optimizer-state hashes
8. record runtime and storage

## Phase E. Identity replay

Replay the WAL with no deletion from the step-zero checkpoint.

Record:

- model hash equality
- optimizer hash equality
- unequal elements
- maximum absolute difference
- replay time

If identity replay fails, deletion replay is not interpreted until the failure is understood.

## Phase F. Deletion oracle and replay

For each deletion scenario:

1. select the forget set
2. expand exact or near-duplicate closure when configured
3. locate the first affected logical step
4. select the latest eligible checkpoint
5. execute an independent slot-preserving deletion oracle
6. physically materialize a redacted token store for main exactness runs
7. replay the WAL against the redacted store
8. run both `slot_mask` and `filter`
9. compare model and optimizer state with the oracle
10. run the repacked counterfactual

## Phase G. Deletion geometry

Use the 410M study for:

- early 0.1 percent
- middle 1 percent
- late 1 percent
- random 5 percent
- canary group deletion
- duplicate closure

Keep scale and geometry as separate axes to control cost.

## Phase H. Model scaling

Run the canonical deletion scenario over:

- 160M
- 410M
- 1B
- 1.4B
- 2.8B if budget permits

Use identical dataset preparation and comparable training semantics wherever architecture allows.

## Phase I. Determinism envelope

Change one factor at a time:

- FP32 versus BF16
- eager versus SDPA
- dropout enabled versus disabled
- strict deterministic algorithms versus relaxed
- optimizer foreach
- optimizer fused
- CPU baseline where feasible
- different GPU architecture if available

Failures stay in the result table.

## Phase J. Provenance sufficiency

Run all provenance ablations from the same base checkpoint:

- RNG seed
- learning rate
- sample order
- microbatch assignment
- accumulation boundary
- logical optimizer step

The paper should describe a provenance field as necessary only when the experiment supports that statement under the tested regime.

## Phase K. Checkpoint tradeoff

Train with a sufficiently fine checkpoint interval, then simulate coarser retention policies. Record:

- retained checkpoint bytes
- nearest eligible checkpoint
- replay distance
- replay wall time
- exactness

This yields a storage-versus-recovery Pareto curve.

## Phase L. Approximate baselines

Run GA, GradDiff, and NPO against the same original model and same deletion request. Use the same audit set.

Do not tune exact replay against approximate methods. Exact replay has no forget-quality hyperparameter because state equality to its oracle is the primary criterion.

## Phase M. Operational extensions

### XOR rollback

Measure patch size, creation time, application time, and exact recovery over consecutive checkpoints.

### Cohort LoRA

Train a cohort adapter over a frozen base, unload it, and compare the recovered base hash with the pre-adapter base hash.

### Curvature hot path

Measure forget and retain behavior before and after the approximate update.

## Phase N. Standard benchmark track

Run the TOFU forget01, forget05, and forget10 studies. Keep benchmark-facing results separate from the WikiText systems matrix.

Use OpenUnlearning or equivalent standard evaluation tooling for standardized external comparisons when the chosen model architecture is supported.

## Phase O. Audits

For exact oracle, replay, repacked retraining, and approximate methods, record relevant secondary audits:

- forget loss
- retain loss
- held-out perplexity
- canary completion likelihood
- greedy canary extraction
- loss-based membership AUC

State equality remains the primary criterion for the exact path.

## Phase P. Aggregate

Run:

```bash
python scripts/aggregate_results.py runs \
  --csv results/experiment_rows.csv \
  --json results/experiment_rows.json
```

Create final machine-readable tables before drafting manuscript prose.

## Phase Q. Release bundle

A release should contain or permanently archive:

- resolved configs
- environment snapshots
- dataset manifests
- execution plans
- WALs
- ordered-ID manifests or a privacy-safe research equivalent
- checkpoint hashes
- final state hashes
- forget-ID lists
- audit JSON
- aggregated tables
- code commit SHA

Large model weights and raw token arrays can be stored separately when repository limits make Git unsuitable, but the archive location must be permanent and cited by the release.

## Phase R. Claims ledger

Populate every evidence cell in `docs/CLAIMS_LEDGER.md`. Delete claims that are not supported.

## Phase S. Paper rewrite

Only after the release artifact is frozen:

1. rewrite the abstract from actual results
2. rewrite the introduction and contributions
3. write the full literature review
4. formalize the trace-preserving counterfactual
5. describe the system from the released code
6. write experimental methodology from frozen configs
7. build tables and figures from aggregated result files
8. write failure modes and limitations from observed evidence
9. keep the paper title unchanged
