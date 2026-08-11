# Baselines and benchmark interoperability

The exact replay contribution is not evaluated only against itself. Comparisons are split into exact counterfactuals, local sanity implementations, and publication-authoritative external baselines.

## Exact counterfactuals

### Trace-preserving deletion oracle

This is the exact target of the replay mechanism. It starts from a checkpoint before the first affected logical optimizer step, follows the original execution plan, and assigns zero loss to forgotten slots.

### Repacked retain-set retraining

This starts from the same eligible checkpoint but densely repacks retained records. It answers a different question because the execution trajectory changes. Both are reported.

## Local sanity implementations

`src/unlearning_at_scale/baselines.py` contains compact implementations of gradient ascent, gradient difference, and NPO. These remain useful for unit tests, objective-level sanity checks, and debugging on the exact same token store.

**They are not the publication-authoritative baseline implementations.** Paper baseline rows should not be sourced from `scripts/run_approximate_baselines.py` unless explicitly labeled as an internal diagnostic.

The curvature hot path is also a project-specific approximate operational experiment and is never presented as exact deletion.

## Publication-authoritative baselines

Publication comparisons use the pinned OpenUnlearning checkout at commit:

`4ad738aaf60f6a4385f6e2506d01da99e76c31f3`

The default TOFU comparison set is:

- GradAscent
- GradDiff
- NPO
- SimNPO

Run them through `scripts/openunlearning_adapter.py tofu-baselines`. The adapter records the upstream commit, target checkpoint, forget split, retain split, methods, and exact commands in a machine-readable manifest.

A direct table comparing replay with approximate methods must use the same chosen target checkpoint and retain reference. The official OpenUnlearning target/retain models are retained as calibration references, not silently mixed with a differently trained local target.

## Standardized evaluation

TOFU evaluation is also run through the pinned OpenUnlearning evaluator with `scripts/openunlearning_adapter.py tofu-eval`. The first-class standardized model is Llama 3.2 1B Instruct. The split mapping is `forget01/retain99`, `forget05/retain95`, and `forget10/retain90`.

MUSE is the budget-dependent extension and is invoked through the same pinned framework after the Pythia and TOFU studies are complete.

## Required reporting

For every approximate method report at least the method, implementation source and commit, exact target checkpoint hash, retain reference, hyperparameters, optimization steps, forget and retain metrics, standardized benchmark metrics, wall-clock time, and resulting model hash.

Approximate methods are never labeled exact because they pass an extraction, membership, or benchmark audit.

## Primary references

- Maini et al., TOFU: A Task of Fictitious Unlearning for LLMs, 2024.
- Shi et al., MUSE: Machine Unlearning Six-Way Evaluation for Language Models, 2024.
- Zhang et al., Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning, 2024.
- Fan et al., Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning, revised 2025.
- Dorna et al., OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics, 2025.
