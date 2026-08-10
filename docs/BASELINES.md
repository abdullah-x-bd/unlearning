# Approximate baselines and benchmark interoperability

The exact replay contribution should not be evaluated only against itself. This repository therefore distinguishes three classes of comparison.

## 1. Exact counterfactuals

### Trace-preserving deletion oracle

This is the exact target of the replay mechanism. It starts from a checkpoint before the first affected logical optimizer step, follows the original execution plan, and assigns zero loss to forgotten slots.

### Repacked retain-set retraining

This starts from the same eligible checkpoint but densely repacks retained records. It answers a different question because the execution trajectory changes.

Both are reported. Neither is silently renamed as the other.

## 2. Approximate optimization baselines

### Gradient ascent

Let `NLL_f` be the mean token negative log likelihood on the forget batch. Gradient ascent on the ordinary language-model loss is implemented by minimizing

`L_GA = -NLL_f`.

### Gradient difference

The retain set is added as an ordinary language-model objective:

`L_GradDiff = -NLL_f + lambda * NLL_r`.

### Negative preference optimization

For a forget sequence `(x, y)`, NPO uses

`L_NPO = -(2 / beta) log sigmoid(-beta * log(p_theta(y|x) / p_ref(y|x)))`.

The implementation precomputes reference sequence NLL values from the original model so a second reference model does not have to remain resident during every NPO update. A retain language-model term is added with an explicit weight.

NPO hyperparameters, optimization steps, batch size, and retain weight must be reported. NPO is approximate even when its empirical audits look strong.

### Curvature hot path

The diagonal-Fisher anti-update is retained as an explicitly approximate operational experiment. It is not grouped with exact replay in result tables.

## 3. Standard benchmark track

TOFU provides a full synthetic QA training set with official forget01, forget05, forget10 and corresponding retain splits. `scripts/prepare_tofu.py` maps those official splits into the same immutable-ID representation used by the replay system.

The broader LLM-unlearning literature now also has OpenUnlearning, which unifies TOFU, MUSE, WMDP, multiple unlearning algorithms, and a larger set of evaluation metrics. The final paper should use that ecosystem where an apples-to-apples standardized benchmark comparison is useful.

The core replay experiment should remain independently reproducible. External benchmark frameworks should therefore be treated as an interoperability track rather than a hidden dependency of the exactness proof.

## Required reporting

For every approximate method report at least:

- method and exact objective
- learning rate
- optimizer
- number of steps
- forget batch size
- retain batch size
- retain weight
- beta when applicable
- forget loss before and after
- retain loss before and after
- held-out utility when available
- membership metric when available
- canary extraction metric when applicable
- wall-clock time
- resulting model hash

Approximate methods should never be labeled exact because they pass an extraction or membership audit.

## Primary references

- Maini et al., TOFU: A Task of Fictitious Unlearning for LLMs, 2024.
- Shi et al., MUSE: Machine Unlearning Six-Way Evaluation for Language Models, 2024.
- Zhang et al., Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning, 2024.
- Fan et al., Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning, revised 2025.
- Dorna et al., OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics, 2025.
