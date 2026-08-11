# External benchmark boundary

The repository deliberately separates novel code from standardized comparison infrastructure.

## Code owned by this project

The paper contribution is implemented in `src/unlearning_at_scale/`: deterministic execution plans, the fixed-width WAL, checkpoint recovery, trace-preserving deletion, redacted replay, state equality checks, provenance ablations, and systems tradeoff measurements.

## Pinned external frameworks

Publication benchmark rows use upstream research implementations fetched by `scripts/bootstrap_upstreams.py` at commits recorded in `external/upstreams.lock.yaml`.

- OpenUnlearning: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- original MUSE implementation: `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e`
- Pythia reference repository: `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`

The upstream repositories are not vendored or modified. They are checked out detached at the pinned commits.

## Study A: controlled systems and scaling

Pythia plus the WikiText trace corpus isolates systems questions: exactness, replay policy, model scale, deletion position, deterministic-kernel sensitivity, checkpoint spacing, WAL cost, and physical redaction. Pythia models use the `step143000` checkpoint as the human-readable source ref; the release lock resolves that ref to a full Hugging Face commit SHA before execution.

## Study B: standardized TOFU benchmark

The first standardized benchmark uses Llama 3.2 1B Instruct and TOFU. `scripts/prepare_tofu_openunlearning.py` reproduces the pinned OpenUnlearning chat template and masks all labels except the final assistant response. The resulting fixed-shape token store can be consumed by the exact replay engine without changing benchmark supervision semantics.

OpenUnlearning is then used for publication evaluation and approximate baselines. Direct comparison tables must use the same chosen target checkpoint and the same retain reference. Our local GA, GradDiff, and NPO implementations are sanity checks only.

The benchmark splits are:

- `forget01` with `retain99`
- `forget05` with `retain95`
- `forget10` with `retain90`

The default publication baselines are GradAscent, GradDiff, NPO, and SimNPO from the pinned upstream checkout.

## Study C: MUSE extension

MUSE is a budget-dependent external validation track after Study A and Study B. It uses the pinned OpenUnlearning MUSE integration and also records the original MUSE repository commit. The planned analyses cover News and Books, deletion-size scaling, and sequential deletion sustainability.

## Artifact freezing

`locks/artifact-sources.yaml` lists every remote model and dataset source. After datasets are prepared, run:

```bash
python scripts/bootstrap_upstreams.py
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

`freeze_artifacts.py` resolves Hub refs to full 40-character commits and hashes every file in each prepared token store. Publication runs use `scripts/run_release.py` or `scripts/run_release_matrix.py`, which refuse to run if the lock no longer verifies.

The generated `locks/artifacts.lock.json` is part of the release evidence and should be committed only after the final datasets are prepared and frozen.
