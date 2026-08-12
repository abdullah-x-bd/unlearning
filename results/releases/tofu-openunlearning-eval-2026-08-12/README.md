# OpenUnlearning TOFU forget10 evaluation release

This directory records the completed standardized OpenUnlearning evaluation of the frozen Llama 3.2 1B Instruct + TOFU `forget10` states produced by the exact-replay release at [`../tofu-llama32-1b-forget10-2026-08-11/`](../tofu-llama32-1b-forget10-2026-08-11/README.md).

The evaluation was run on 2026-08-12 with the pinned OpenUnlearning checkout at commit `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`. The canonical original and trace-preserving deletion states were reconstructed from the frozen release evidence, hash-verified against the release record, exported to Hugging Face format, and evaluated without any additional training.

## Status

**Passed.** The evaluation, evidence handoff, local manifest verification, artifact upload, Pod cleanup, metadata upload, and final fail-closed workflow gate all completed successfully.

The evaluation-only recovery metadata records:

- training passes: `0`
- optimizer updates: `0`
- GPU: NVIDIA A40
- PyTorch: `2.4.1+cu124`
- CUDA runtime: `12.4`
- attention implementation: `eager`
- reported GPU price: `$0.44/hour`
- elapsed time through cleanup: `2410.49 s`
- estimated GPU charge through cleanup: `$0.2946`

## Evaluated states

The reconstruction reused the frozen canonical hashes from the Aug 11 exactness release.

| State | Model SHA-256 | Optimizer SHA-256 |
| --- | --- | --- |
| Original full-data target | `54c711e9bde77215d9c5def50429f925a382bdcd28150bb87a89a118dd54bc65` | `7dae55302988b834099f224a6ababcbd97a3411afa1b54e5e9a1739854f8e773` |
| Trace-preserving `forget10` deletion | `067109bfd2e34f1616a8069d04ecd28b4814513332b03957ab917503122aeec3` | `0152484534eab198ebee5a500e77d81451f2cba34ea6baa987ccc5726e20daa1` |

The deletion state corresponds to the physically redacted replay store in which 400 of 4,000 TOFU records were removed, 3,600 records remained, benchmark label masks were preserved, and `forgotten_ids_present` was false.

## Standardized TOFU results

The official OpenUnlearning `retain90` checkpoint was pinned to Hugging Face revision `7114300c0049527a71833f5683965c358ad9dcbf` and used as the immutable retain reference required by the evaluator.

| Metric | Original target | Exact deletion | Official `retain90` reference |
| --- | ---: | ---: | ---: |
| Model Utility | 0.3542886523 | **0.3541358896** | 0.5927319186 |
| Forget Quality | 5.974773e-08 | **3.913230e-08** | n/a |
| Forget Truth Ratio | 0.7502411073 | **0.7504341644** | 0.6275970313 |
| Forget QA probability | 0.1682556152 | **0.1669091797** | 0.1161531067 |
| Forget ROUGE | 0.3445035978 | **0.3475321272** | 0.3796347366 |
| Retain QA probability | 0.1650482178 | **0.1646966553** | 0.8802783203 |
| Retain ROUGE | 0.3222409301 | **0.3228803928** | 0.8282506403 |
| Retain Truth Ratio | 0.1947497154 | **0.1954338163** | 0.5130911416 |
| Min-K MIA AUC | 0.3398906250 | **0.3368312500** | 0.3823312500 |
| Extraction strength | 0.0573743476 | **0.0568245281** | 0.0596114221 |
| PrivLeak | 6.8710979782 | **7.3664079642** | 23.5337499953 |

The standardized evaluation therefore records a clear empirical pattern in this setting: the exact trace-preserving deletion state remains very close to the original full-data target on the core utility, probability, ROUGE, MIA, and extraction aggregates, while its TOFU Forget Quality remains extremely small relative to the official `retain90` reference.

## Interpretation boundary

This release supports the following narrow behavioral claim:

> For the frozen Llama 3.2 1B + TOFU `forget10` experiment, a state-exact trace-preserving deletion counterfactual can remain behaviorally close to the original target under the pinned OpenUnlearning evaluation while scoring very differently from the official `retain90` calibration reference.

This result is important, but its interpretation is deliberately bounded.

- The official `retain90` checkpoint is a separately trained calibration reference, not a matched local retain-only retrain from this project's exact target-training pipeline.
- Therefore this release does **not** by itself prove that the observed behavioral difference is caused solely by trace preservation versus dense retain-only repacking.
- A matched local retain-only control is required before making that causal comparison.
- No superiority claim is made over GradAscent, GradDiff, NPO, SimNPO, or other approximate methods. Publication-authoritative approximate baseline runs remain separate work.
- The result does not establish multi-GPU exactness, arbitrary-stack exactness, privacy guarantees, or legal compliance.

## OpenUnlearning provenance

- framework: OpenUnlearning
- framework commit: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- benchmark: TOFU
- forget split: `forget10`
- retain split: `retain90`
- holdout split: `holdout10`
- model config: `Llama-3.2-1B-Instruct`
- attention implementation: `eager`
- official retain reference: `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`
- retain reference revision: `7114300c0049527a71833f5683965c358ad9dcbf`
- canonical frozen source run: `31490644488`
- canonical frozen source commit: `c2375c1e491224062c05de6a3abe1b50f4af3937`
- canonical source evidence artifact ID: `9104982056`
- recovery checkpoint artifact ID: `9114652333`

## Evaluation workflow evidence

- GitHub Actions workflow run: `31607929368`
- GitHub Actions job: `94151588320`
- evaluated repository head: `ce99971f0bc6545164c232789e317ff1e575c950`
- verified results artifact: `tofu-openunlearning-results`
- results artifact ID: `9147513923`
- results artifact ZIP SHA-256: `b143e1784111ae3ae1f8ae7ef3bcc4d1502de2f76a3dd2e0254c311332ecacdd`
- internal results TAR SHA-256: `d6b82c231fb73fecfa12996a4eb5bc68cdbb93959989ed660d7351464a6ce4ae`
- recovery metadata artifact: `tofu-openunlearning-recovery-metadata`
- metadata artifact ID: `9147515241`
- metadata artifact ZIP SHA-256: `9f795ef861b03d53d72a4856464bd6d4c9393c1f10fabefe5e3be6729cc30d5d`

The result artifact contains the three evaluator logs (`original`, `deletion`, and `retain90`), interoperability command manifests, reconstruction summary, frozen hashes, GPU probe, allocation metadata, BF16 interoperability patch provenance, and a handoff manifest. The workflow re-parsed every JSON file and rechecked its exact byte size and SHA-256 before the artifact could be uploaded.

## Durable machine-readable summary

See [`summary.json`](summary.json) for the compact machine-readable release record. The larger evaluator logs remain available in the GitHub Actions result artifact and should also be attached to the corresponding GitHub Release so that the evidence does not depend only on Actions retention.
