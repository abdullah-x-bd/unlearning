# External benchmark boundary

The repository deliberately separates novel project code from standardized comparison infrastructure.

## Code owned by this project

The paper contribution is implemented in `src/unlearning_at_scale/`: deterministic execution plans, the fixed-width WAL, checkpoint recovery, trace-preserving deletion, redacted replay, state equality checks, provenance ablations, and systems tradeoff measurements.

## Pinned external frameworks

Publication benchmark rows use upstream research implementations fetched by `scripts/bootstrap_upstreams.py` at commits recorded in `external/upstreams.lock.yaml`.

- OpenUnlearning: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- original MUSE implementation: `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e`
- Pythia reference repository: `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`

Upstream repositories are checked out detached at the pinned commits. The OpenUnlearning evaluation path applies one narrow, recorded interoperability patch in `src/evals/metrics/utils.py`: BF16 metric tensors are cast to float32 immediately before NumPy conversion. BF16 values are exactly representable in float32, so this changes representation compatibility rather than metric values or benchmark logic. The evaluation artifact records the original file hash, patched file hash, exact two replacements, purpose, and upstream commit in `bf16_numpy_patch.json`.

No project-specific unlearning objective or metric implementation is substituted for the pinned OpenUnlearning evaluator.

## Study A: controlled systems and scaling

Pythia plus the WikiText trace corpus isolates systems questions: exactness, replay policy, model scale, deletion position, provenance cost, and physical redaction. The completed publication-scale endpoints are Pythia 160M and Pythia 2.8B. Pythia models use the `step143000` checkpoint as the human-readable source ref; the release lock resolves that ref to a full Hugging Face commit SHA before execution.

## Study B: standardized TOFU benchmark

The standardized benchmark-facing study uses Llama 3.2 1B Instruct and TOFU. `scripts/prepare_tofu_openunlearning.py` reproduces the pinned OpenUnlearning chat template and masks all labels except the final assistant response. The resulting fixed-shape token store can be consumed by the exact replay engine without changing benchmark supervision semantics.

The frozen exactness release is `forget10` with `retain90` semantics:

- full target records: 4,000
- requested deletion records: 400
- physically redacted replay records: 3,600
- exact identity replay: passed
- exact trace-preserving deletion replay: passed
- optimizer equality: passed
- forgotten IDs present after redaction: false

The canonical frozen original and deletion states were subsequently reconstructed from release evidence, hash-verified, exported to Hugging Face format, and evaluated through OpenUnlearning on 2026-08-12. The evaluation used:

- forget split: `forget10`
- retain split: `retain90`
- holdout split: `holdout10`
- attention implementation: `eager`
- official retain reference: `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`
- retain reference revision: `7114300c0049527a71833f5683965c358ad9dcbf`

The exact deletion checkpoint remained close to the project's frozen original target on the core Model Utility, QA probability, ROUGE, Min-K MIA, and extraction aggregates. Its TOFU Forget Quality was `3.913229651378406e-08`, compared with `5.974773347845895e-08` for the original target, when each was evaluated against the official `retain90` reference.

The durable evaluation record is [`results/releases/tofu-openunlearning-eval-2026-08-12/`](../results/releases/tofu-openunlearning-eval-2026-08-12/README.md).

### Comparison boundary

Direct comparison tables must distinguish three objects:

1. the project's frozen original target,
2. the project's trace-preserving exact deletion counterfactual,
3. a retain-only or approximate baseline trained from a specified target and protocol.

The official OpenUnlearning `retain90` checkpoint is retained as a calibration reference. It is not silently treated as a matched local retain-only retrain of this project's target. Therefore the completed standardized evaluation is a valid benchmark measurement, but it does not by itself identify the causal effect of trace preservation versus dense repacking.

A matched local retain-only control is required before making that causal comparison central to the manuscript.

## Publication-authoritative approximate baselines

The default comparison set remains:

- GradAscent
- GradDiff
- NPO
- SimNPO

These must be run from the pinned OpenUnlearning checkout through `scripts/openunlearning_adapter.py tofu-baselines`. Our local GA, GradDiff, and NPO implementations are sanity checks only and do not supply publication-authoritative baseline rows.

No superiority claim over these methods is currently supported by the frozen evidence.

## Study C: MUSE extension

MUSE remains a budget-dependent external validation track. It uses the pinned OpenUnlearning MUSE integration and records the original MUSE repository commit. The planned analyses cover News and Books, deletion-size scaling, and sequential deletion sustainability.

## Artifact freezing

`locks/artifact-sources.yaml` lists every remote model and dataset source. After datasets are prepared, run:

```bash
python scripts/bootstrap_upstreams.py
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

`freeze_artifacts.py` resolves Hub refs to full 40-character commits and hashes every file in each prepared token store. Publication runs use `scripts/run_release.py` or `scripts/run_release_matrix.py`, which refuse to run if the lock no longer verifies.

The generated `locks/artifacts.lock.json` is part of the release evidence. Changing locked data or upstream revisions requires a new release identity.
