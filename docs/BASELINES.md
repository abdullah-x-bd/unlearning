# Baselines and benchmark interoperability

The exact replay contribution is not evaluated only against itself. Comparisons are split into exact counterfactuals, local sanity implementations, standardized calibration references, and publication-authoritative external baselines.

## Exact counterfactuals

### Trace-preserving deletion oracle

This is the exact target of the replay mechanism. It starts from a checkpoint before the first affected logical optimizer step, follows the original execution plan, and assigns zero loss to forgotten slots while preserving their logical positions in the trace.

### Repacked retain-set retraining

This starts from the same eligible checkpoint but densely repacks retained records. It answers a different systems counterfactual because the execution trajectory changes. In the released Pythia 160M study, repacked retraining diverged from the trace-preserving deletion oracle in all four tested deletion scenarios.

A matched local retain-only TOFU control has **not** yet been run for the frozen Llama 3.2 1B target. Until it is run, the official OpenUnlearning `retain90` checkpoint must not be described as if it were a matched local retrain of the project's target.

## Local sanity implementations

`src/unlearning_at_scale/baselines.py` contains compact implementations of gradient ascent, gradient difference, and NPO. These remain useful for unit tests, objective-level sanity checks, and debugging on the exact same token store.

**They are not the publication-authoritative baseline implementations.** Paper baseline rows should not be sourced from `scripts/run_approximate_baselines.py` unless explicitly labeled as an internal diagnostic.

The curvature hot path is also a project-specific approximate operational experiment and is never presented as exact deletion.

## Standardized OpenUnlearning evaluation

The frozen Llama 3.2 1B original and trace-preserving `forget10` deletion states have now been evaluated through the pinned OpenUnlearning checkout at commit:

`4ad738aaf60f6a4385f6e2506d01da99e76c31f3`

The evaluation used the official `retain90` checkpoint pinned to revision:

`7114300c0049527a71833f5683965c358ad9dcbf`

The evaluation completed successfully and is frozen in [`results/releases/tofu-openunlearning-eval-2026-08-12/`](../results/releases/tofu-openunlearning-eval-2026-08-12/README.md).

The observed pattern is that the exact deletion state remains close to the original target on the core standardized behavioral aggregates while both have extremely small TOFU Forget Quality values against the official `retain90` reference. This is a valid benchmark result, but the official reference is a separately trained calibration model. A matched local retain-only control is required before attributing the behavioral separation causally to trace preservation versus dense repacking.

## Publication-authoritative approximate baselines

Publication comparisons use the same pinned OpenUnlearning checkout. The default TOFU comparison set is:

- GradAscent
- GradDiff
- NPO
- SimNPO

Run them through `scripts/openunlearning_adapter.py tofu-baselines`. The adapter records the upstream commit, target checkpoint, forget split, retain split, methods, and exact commands in a machine-readable manifest.

For any direct method-comparison table:

1. choose one explicit target checkpoint,
2. use that same target for every approximate method,
3. use one explicitly identified retain reference,
4. record hyperparameters and optimization steps,
5. preserve exact output logs and resulting model hashes.

The publication-authoritative approximate baseline runs are **not yet part of the frozen evidence set**, so no superiority claim is made over GradAscent, GradDiff, NPO, SimNPO, or other methods.

## Standardized evaluation protocol

TOFU evaluation is run through `scripts/openunlearning_adapter.py tofu-eval`. The first-class standardized model is Llama 3.2 1B Instruct. The supported split mapping is `forget01/retain99`, `forget05/retain95`, and `forget10/retain90`; the current frozen benchmark-facing release is `forget10/retain90`.

The OpenUnlearning path applies a recorded BF16-to-float32 NumPy interoperability cast immediately before NumPy conversion. The patch does not alter the BF16 metric values or evaluator logic and is frozen in `bf16_numpy_patch.json` with before/after hashes and exact replacements.

MUSE is the budget-dependent extension and is invoked through the same pinned framework if pursued after the core paper evidence is frozen.

## Required reporting

For every approximate method report at least the method, implementation source and commit, exact target checkpoint hash, retain reference, hyperparameters, optimization steps, forget and retain metrics, standardized benchmark metrics, wall-clock time, and resulting model hash.

Approximate methods are never labeled exact because they pass an extraction, membership, or benchmark audit.

## Primary references

- Maini et al., TOFU: A Task of Fictitious Unlearning for LLMs, 2024.
- Shi et al., MUSE: Machine Unlearning Six-Way Evaluation for Language Models, 2024.
- Zhang et al., Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning, 2024.
- Fan et al., Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning, revised 2025.
- Dorna et al., OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics, 2025.
