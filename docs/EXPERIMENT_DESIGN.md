# Experimental design

## Primary research question

Can language-model training be engineered so that a later deletion request can be executed as a reproducible counterfactual computation?

The project studies unlearning as a training-systems problem rather than only as a post-training optimization problem.

## Counterfactual definitions

### Trace-preserving deletion

The original execution plan fixes microbatch slots, per-microbatch RNG seeds, logical optimizer-step boundaries, and learning-rate values. A forget request removes selected records from the loss while preserving the rest of that execution plan.

### Repacked retain-set retraining

Retained records are densely packed into new batches and a normal schedule is rebuilt. This is an important comparison, but it is not assumed to be byte equivalent to the trace-preserving target.

The paper must distinguish these targets explicitly.

## Research questions

### RQ1. Can the original trace be reconstructed?

Test whether the fixed-width WAL plus ordered-ID manifest reconstructs the execution plan exactly.

### RQ2. Can no-deletion replay reproduce original training?

Run identity replay from the step-zero checkpoint and compare model and optimizer states.

### RQ3. Can deletion replay reproduce an independent deletion oracle?

For each forget request, run the oracle and replay independently from the same eligible checkpoint.

### RQ4. Can replay operate after physical deletion?

Materialize a redacted token store that omits the forgotten rows before replay.

### RQ5. Which replay policy is more robust?

Compare slot masking with physical row filtering.

### RQ6. Which provenance fields matter empirically?

Corrupt one recorded control variable at a time and measure state divergence.

### RQ7. Where does determinism break?

Vary dtype, kernels, dropout, optimizer implementation, device, and eventually distributed execution.

### RQ8. How does deletion cost scale?

Measure replay latency against model size, deletion position, forget-set geometry, and checkpoint distance.

### RQ9. What are the storage tradeoffs?

Measure WAL, manifest, checkpoint, rollback-patch, and adapter storage.

### RQ10. How does exact replay compare with approximate unlearning?

Run GA, GradDiff, NPO, and the curvature hot path against the same deletion requests and audits.

## Primary model scale

| Model | Role |
| --- | --- |
| Pythia 160M | low-cost full pipeline |
| Pythia 410M | deletion-geometry anchor |
| Pythia 1B | billion-parameter regime |
| Pythia 1.4B | primary large run |
| Pythia 2.8B | budget-dependent scale extension |

The tiny CI model is never used as scientific evidence.

## Primary dataset

WikiText-103 is converted to fixed-length causal-LM records. The preparation step adds:

- immutable sample IDs
- content SHA-256 values
- deterministic SimHash signatures
- fixed token arrays
- synthetic canary groups
- dataset manifest and hashes

## Standard benchmark track

TOFU is prepared separately using its official full, forget01, forget05, forget10, retain99, retain95, and retain90 sets. This provides a benchmark-facing validation track without making the core replay engine depend on an external benchmark implementation.

## Forget-set axes

### Geometry study on 410M

- early 0.1 percent
- middle 1 percent
- late 1 percent
- random 5 percent
- controlled canary group

### Scaling study

Use a canonical late 1 percent deletion across model sizes.

### Duplicate study

Test:

- exact content-hash closure
- SimHash near-duplicate closure at reported Hamming thresholds

### TOFU study

Use official forget01, forget05, and forget10 ID sets.

## Main replay policies

### Slot mask

Retain original batch shape. Forgotten IDs become dummy-token slots with zero loss weight.

### Filter

Remove forgotten rows before the forward pass.

### Repacked

Densely repack retained IDs into a new training trajectory.

## Exactness metrics

Record:

- full model SHA-256
- full optimizer SHA-256
- unequal tensor count
- unequal element count
- maximum absolute parameter difference
- L2 parameter difference
- logical steps
- applied updates
- skipped updates
- wall time

## Systems metrics

Record:

- WAL bytes
- manifest bytes
- total provenance bytes
- checkpoint bytes
- checkpoint retention bytes
- replay distance
- rollback patch bytes
- adapter bytes
- training and replay time

## Secondary audits

For exact paths, state equality is stronger than behavioral auditing. Secondary audits remain useful for approximate methods and cross-counterfactual interpretation:

- forget loss
- retain loss
- held-out perplexity
- loss-based membership inference AUC
- canary completion NLL
- greedy canary extraction

## Provenance ablations

Corrupt exactly one trace component at a time:

- per-microbatch seed
- learning rate
- sample order
- microbatch assignment
- accumulation boundary
- logical optimizer-step value

The resulting divergence maps the empirical sufficiency envelope of the trace.

## Determinism stress suite

Change one factor at a time:

- FP32 versus BF16
- eager attention versus SDPA
- deterministic algorithms strict versus relaxed
- dropout enabled versus disabled
- optimizer foreach versus scalar implementation
- optimizer fused versus unfused
- CPU versus GPU where feasible
- checkpoint-resume distance
- microbatch size
- accumulation length

A separate multi-GPU study is required before making distributed exactness claims.

## Approximate baselines

Use:

- gradient ascent
- gradient difference
- NPO
- curvature anti-update

For standardized external comparison, use the TOFU/OpenUnlearning ecosystem where appropriate and keep its results clearly separated from the systems-scale matrix.

## Operational extensions

### XOR rollback

Build bytewise XOR patches between consecutive checkpoint states and measure exact recovery, storage, creation time, and application time.

### Cohort LoRA

Train cohort-scoped adapters over a frozen base and test exact base hash recovery after adapter unload.

### Curvature hot path

Use the diagonal-Fisher update as a temporary approximate path and evaluate forget/retain damage.

## Result policy

Failure cases remain in the released tables. A configuration that is numerically close but not byte exact must not be labeled exact.

The manuscript should be written from frozen result artifacts after the full runbook is complete.
