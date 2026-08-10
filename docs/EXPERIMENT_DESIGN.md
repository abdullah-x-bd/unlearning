# Experimental design

## Research question

The primary question is not whether a post-hoc objective can make a model appear to forget. It is whether training can be engineered so that a later deletion request can be executed as a reproducible counterfactual computation.

The core target is a **trace-preserving deletion counterfactual**. The original execution plan fixes microbatch slots, optimizer-step boundaries, random seeds, and the learning-rate value attached to each logical step. A deletion removes a record's contribution to the loss while leaving this execution plan fixed.

This is intentionally distinct from a fresh run that repacks `D \\ F` into new batches. Repacking changes grouping, random-number consumption, optimizer-step boundaries, and often the learning-rate trajectory. That alternative is measured as a diagnostic baseline rather than silently treated as the same target.

## Main hypotheses

### H1 WAL sufficiency

A fixed-width training WAL plus an immutable batch manifest is sufficient to reconstruct the trace-preserving execution plan from a clean checkpoint.

### H2 exact replay

With deterministic algorithms, an exact checkpoint, sum-reduced token loss, and a fixed execution plan, replay using the reconstructed WAL matches an independently executed trace-preserving oracle in model parameters. Optimizer-state equality is recorded separately.

### H3 slot preservation improves numerical robustness

Replacing forgotten slots with deterministic dummy tokens and multiplying their token losses by zero should preserve batch shapes and random-number allocation better than physically shrinking microbatches. The study compares both policies.

### H4 repacked retraining is a different counterfactual

Ordinary retain-set repacking will generally diverge from the trace-preserving counterfactual. The magnitude and practical significance of this divergence are empirical questions.

### H5 exactness has a systems envelope

Byte equality may depend on dtype, attention implementation, optimizer implementation, hardware, dropout, distributed collectives, and software versions. Failures are recorded as findings, not hidden.

## Model scale

The main matrix uses the Pythia family because the models share a research-oriented architecture and span multiple parameter scales.

| Model | Role |
| --- | --- |
| Pythia 160M | lower-cost full pipeline |
| Pythia 410M | small nontrivial scale |
| Pythia 1B | billion-parameter regime |
| Pythia 1.4B | primary large run |
| Pythia 2.8B | budget-dependent scale extension |

The tiny model in CI is only a software regression test. It is never used as scientific evidence.

## Data

The default preparation script samples WikiText-103 and creates fixed-length token records. Synthetic canary groups are injected only for controlled memorization and extraction audits. Every prepared dataset receives immutable sample IDs, content hashes, token arrays, and a preparation manifest.

No large dataset or model artifact is committed to Git.

## Forget-set axes

The scale matrix uses a canonical late 1 percent deletion across model sizes. A separate 410M deletion-geometry study tests early 0.1 percent, middle 1 percent, late 1 percent, and random 5 percent. This avoids multiplying every geometric condition by every expensive model size. Additional release runs should add canary-group deletion, duplicate closure, multiple random seeds, and checkpoint-distance sweeps.

## Replay methods

1. `slot_mask` keeps original microbatch shape, substitutes a dummy sequence for a forgotten ID, and multiplies that slot's token loss by zero.
2. `filter` removes forgotten rows before the forward pass.
3. `repacked` rebuilds dense retain-only microbatches and recomputes a normal schedule. It is not the exact target.

## Exactness metrics

For every oracle/replay pair, record:

- full state SHA-256
- number of unequal tensors
- number of unequal elements
- maximum absolute parameter difference
- L2 parameter difference
- optimizer-state SHA-256
- applied and skipped optimizer updates
- wall time
- WAL bytes
- checkpoint bytes

## Unlearning audits

State equality to the counterfactual oracle is the strongest audit for the exact path. Secondary audits are still useful for approximate methods and for comparison with the repacked baseline:

- forget loss
- retain loss
- held-out perplexity
- canary continuation likelihood
- extraction success
- loss-based membership inference AUC
- exact and content-hash duplicate closure

## Stress tests

The determinism suite changes one factor at a time:

- CPU versus single GPU
- FP32 versus BF16
- eager attention versus SDPA
- deterministic algorithms on/off
- native dropout versus disabled dropout
- optimizer foreach/fused variants
- checkpoint resume distance
- microbatch size and accumulation length

Distributed FSDP/TP claims are not made until a separate multi-GPU suite is implemented and run.

## Operational extensions

The repository retains two extensions from the original research idea but keeps them secondary to the exact replay result.

### Recent exact rollback

`rollback.py` constructs XOR byte patches over tensors and stores prior scalar metadata. This is evaluated for exact restoration, storage cost, and creation/application latency.

### Cohort adapters

`lora.py` supports cohort-scoped LoRA adapters over a frozen base. Deleting a cohort adapter recovers the unchanged base model by construction. Experiments should measure adapter training utility and the storage/latency tradeoff.

### Approximate hot path

`hotpath.py` implements a diagonal-Fisher curvature anti-update. It is explicitly approximate and must be evaluated against the same audits. It must never be reported as exact deletion.
