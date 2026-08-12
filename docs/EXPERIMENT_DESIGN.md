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

**Frozen evidence:** supported at Pythia 160M, Pythia 2.8B, and Llama 3.2 1B + TOFU `forget10`.

### RQ2. Can no-deletion replay reproduce original training?

Run identity replay from the step-zero checkpoint and compare model and optimizer states.

**Frozen evidence:** exact at all three publication-facing model endpoints.

### RQ3. Can deletion replay reproduce an independent deletion oracle?

For each forget request, run the oracle and replay independently from the same eligible checkpoint.

**Frozen evidence:** exact at Pythia 160M across four deletion geometries, Pythia 2.8B random 5%, and Llama 3.2 1B TOFU `forget10`.

### RQ4. Can replay operate after physical deletion?

Materialize a redacted token store that omits the forgotten rows before replay.

**Frozen evidence:** yes in every released deletion experiment.

### RQ5. Which replay policy is more robust?

Compare slot masking with physical row filtering.

**Frozen evidence:** the Pythia 160M systems study shows exact slot-mask replay and non-exact filter replay in all four frozen scenarios. This comparison was intentionally not repeated at every larger scale.

### RQ6. Which provenance fields matter empirically?

Corrupt one recorded control variable at a time and measure state divergence.

**Status:** implementation exists; publication-scale ablation runs are not in the current frozen evidence set.

### RQ7. Where does determinism break?

Vary dtype, kernels, dropout, optimizer implementation, device, and eventually distributed execution.

**Status:** selected environment checks exist; a complete determinism envelope and multi-GPU study remain outside the current frozen claim set.

### RQ8. How does deletion cost scale?

Measure replay latency against model size, deletion position, forget-set geometry, and checkpoint distance.

**Frozen evidence:** direct exactness scale check from Pythia 160M to Pythia 2.8B, with release-specific runtime/provenance measurements. A full checkpoint-distance Pareto sweep remains optional.

### RQ9. What are the storage tradeoffs?

Measure WAL, manifest, checkpoint, rollback-patch, and adapter storage.

**Frozen evidence:** WAL and ordered-ID manifest costs are reported for all three publication-facing exactness releases. Broader rollback/adapter economics remain optional extensions.

### RQ10. How does exact replay compare with approximate unlearning?

Use publication-authoritative OpenUnlearning implementations of GradAscent, GradDiff, NPO, and SimNPO from one common target when making a direct comparison.

**Status:** integration complete; publication-authoritative comparison runs are not yet frozen, so no superiority claim is made.

### RQ11. How does a state-exact deletion counterfactual score under standardized behavioral unlearning evaluation?

Evaluate the hash-verified frozen original and exact-deletion checkpoints using one pinned OpenUnlearning configuration and preserve the full evaluator logs and provenance.

**Frozen evidence:** complete for Llama 3.2 1B + TOFU `forget10`. The exact deletion state remains close to the project's original target on the core behavioral aggregates while both have extremely small Forget Quality values against the official pinned `retain90` calibration reference.

The official `retain90` checkpoint is not a matched local retain-only retrain of the project's target. A matched local retain-only control is required before attributing that behavioral separation causally to trace preservation versus dense repacking.

## Frozen publication-facing model endpoints

| Model | Role | Frozen result |
| --- | --- | --- |
| Pythia 160M | controlled systems and deletion-geometry endpoint | exact identity replay; exact deletion replay for early 0.1%, middle 1%, late 1%, random 5%; filter and repacked controls |
| Pythia 2.8B | direct large-scale systems check | exact identity replay; exact random-5% deletion replay after physical redaction |
| Llama 3.2 1B Instruct | cross-family standardized benchmark endpoint | exact identity replay; exact TOFU `forget10` deletion replay after physical redaction; completed OpenUnlearning evaluation |

Earlier planning configs for intermediate Pythia sizes remain in the repository as experimental infrastructure, but they are not presented as completed publication evidence.

The tiny CI model is never used as scientific evidence.

## Primary systems dataset

WikiText-103 is converted to fixed-length causal-LM records. The preparation step adds:

- immutable sample IDs
- content SHA-256 values
- deterministic SimHash signatures
- fixed token arrays
- synthetic canary support
- dataset manifest and hashes

## Standard benchmark track

TOFU is prepared separately with benchmark-faithful chat formatting and answer-only supervision. The preparation infrastructure supports `forget01`, `forget05`, and `forget10` split files, but the current frozen cross-family exactness and behavioral release uses **`forget10`**.

The frozen Llama experiment uses:

- 4,000 full-set records
- 400 `forget10` records
- 3,600 retained records after physical redaction
- five epochs
- microbatch size 8
- gradient accumulation 4
- 625 logical optimizer updates
- BF16
- eager attention
- strict deterministic algorithms

## Frozen forget-set axes

### Pythia 160M geometry study

- early 0.1%
- middle 1%
- late 1%
- random 5%

### Pythia 2.8B scale study

- random 5%

### TOFU cross-family study

- `forget10` / 400 of 4,000 records

Additional canary, duplicate-closure, intermediate-size, `forget01`, and `forget05` studies remain available as extensions but are not part of the first frozen claim set.

## Main replay policies

### Slot mask

Retain original logical slots. Forgotten IDs contribute zero loss while retained computation follows the frozen execution trace.

### Filter

Remove forgotten rows before the forward pass. This can alter batch shape and execution details and is treated as a distinct replay policy.

### Repacked

Densely repack retained IDs into a new training trajectory. This is a separate counterfactual, not the definition of trace-preserving deletion.

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
- checkpoint bytes where relevant
- replay distance where relevant
- training and replay time
- GPU/environment provenance
- artifact hashes and workflow identifiers

## Standardized behavioral metrics

For the TOFU/OpenUnlearning track preserve the full evaluator logs and report at least:

- Forget Quality
- Model Utility
- forget and retain QA probability
- forget and retain ROUGE
- Truth Ratio statistics
- Min-K membership-inference statistic
- extraction strength
- PrivLeak

State equality and behavioral metrics answer different questions and must not be conflated.

## Secondary audits

For exact paths, state equality is stronger than behavioral similarity as an exactness test. Secondary audits remain useful for approximate methods and cross-counterfactual interpretation:

- forget loss
- retain loss
- held-out perplexity
- loss-based membership inference AUC
- canary completion NLL
- greedy canary extraction

## Provenance ablations

The implementation can corrupt exactly one trace component at a time:

- per-microbatch seed
- learning rate
- sample order
- microbatch assignment
- accumulation boundary
- logical optimizer-step value

A field is described as empirically necessary only after the corresponding frozen ablation evidence exists.

## Determinism stress suite

Potential one-factor changes include:

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

For standardized external comparison, use publication-authoritative implementations from the pinned OpenUnlearning checkout:

- GradAscent
- GradDiff
- NPO
- SimNPO

Project-local baseline implementations remain diagnostic only. Direct comparison requires one common target and an explicitly identified retain reference.

## Operational extensions

### XOR rollback

Build bytewise XOR patches between consecutive checkpoint states and measure exact recovery, storage, creation time, and application time.

### Cohort LoRA

Train cohort-scoped adapters over a frozen base and test exact base hash recovery after adapter unload.

### Curvature hot path

Use the diagonal-Fisher update as a temporary approximate path and evaluate forget/retain damage.

These extensions are not prerequisites for the first frozen exactness and OpenUnlearning release.

## Result policy

Failure cases remain in released evidence. A configuration that is numerically close but not byte exact must not be labeled exact.

No value moves into the manuscript merely because it appeared in a console log. Publication claims are sourced from frozen release artifacts and the durable records under `results/releases/`.

The current claim boundary is authoritative in `docs/CLAIMS_LEDGER.md`.
