# Claims ledger

No candidate claim moves into the rebuilt manuscript until its required evidence is present in a released artifact.

| ID | Candidate claim | Required evidence | Current status |
| --- | --- | --- | --- |
| G1 | WAL reconstruction reproduces the intended execution plan | plan/WAL equality across release runs | supported at Pythia 160M, Pythia 2.8B, and Llama 3.2 1B + TOFU forget10; all three successful releases passed the WAL-to-plan equality guard |
| G2 | no-deletion replay can reproduce the original run | model and optimizer identity replay across release matrix | exact model and optimizer identity at Pythia 160M, Pythia 2.8B, and Llama 3.2 1B + TOFU forget10; additional seeds remain untested |
| G3 | trace-preserving deletion replay can be byte exact | oracle versus replay hashes and tensor comparisons across scales and model families | exact at Pythia 160M for early 0.1%, middle 1%, late 1% and random 5%; exact at Pythia 2.8B for random 5%; exact at Llama 3.2 1B for TOFU forget10; additional seed replication remains untested |
| G4 | physical deletion of token rows is compatible with replay | replay against materialized redacted stores | supported at Pythia 160M across four scenarios, Pythia 2.8B random 5%, and Llama 3.2 1B TOFU forget10; forgotten IDs were absent from every tested materialized replay store |
| G5 | slot masking is more robust than physical filtering | determinism and deletion matrix | Pythia 160M release shows exact slot-mask replay and non-exact filter replay in all four scenarios; filter was intentionally not repeated at 2.8B or on the focused TOFU release |
| G6 | repacked retain-only retraining can diverge from the trace-preserving target | same-checkpoint repacked comparisons | supported at Pythia 160M in all four release scenarios; repacked retraining was intentionally not repeated at 2.8B or on the focused TOFU release |
| G7 | specific provenance fields are empirically necessary | one-field provenance ablations | implementation complete, runs pending |
| G8 | checkpoint spacing creates a storage versus replay-latency tradeoff | checkpoint interval sweep | implementation complete, runs pending |
| G9 | exact and near-duplicate closure changes deletion scope | closure counts and downstream experiments | implementation complete, runs pending |
| G10 | XOR rollback restores earlier state bytes exactly | unit tests plus scaled checkpoint benchmark | regression test passes, scale benchmark pending |
| G11 | cohort adapter deletion can recover the unchanged base exactly | base hash before adapter and after unload | implementation complete, GPU run pending |
| G12 | curvature hot path provides useful approximate deletion behavior | forget/retain/audit comparison against baselines | implementation complete, runs pending |
| G13 | replay compares with established approximate methods on standardized unlearning metrics | same-target GradAscent, GradDiff, NPO and SimNPO runs from pinned OpenUnlearning plus exact command manifests | integration complete, baseline runs pending; no comparative claim yet |
| G14 | the exact replay mechanism generalizes beyond the controlled Pythia/WikiText systems study | cross-family Llama + TOFU exact state evidence, followed by standardized behavioral evaluation | state-level generalization supported by exact Llama 3.2 1B TOFU forget10 identity and physically-redacted deletion replay; pinned OpenUnlearning behavioral evaluation is pending |
| G15 | results are reproducible from frozen external dependencies and data | verified artifact lock containing full Hub SHAs, external Git commits and prepared-dataset file hashes | frozen artifact lock verified; CPU dataset preparation reproduced byte-for-byte across clean runners; successful Pythia 160M, Pythia 2.8B, and TOFU Llama 1B GPU releases executed behind the verified release-lock gate |
| G16 | MUSE confirms behavior under larger external validation including scaling/sequential deletion | pinned MUSE News/Books results | budget-dependent, no claim yet |
| G17 | distributed replay is exact | dedicated multi-GPU deterministic suite | not implemented, claim prohibited |
| G18 | the mechanism directly satisfies a specific legal erasure obligation | separate legal analysis tied to actual technical guarantees | not a technical claim |

Released empirical evidence:

- `results/releases/pythia-160m-2026-08-11/`: GitHub Actions artifact `9088115288`, digest `sha256:f643f9baea2e819ac2494b605d11ee3dc15d991d9e1bd9bc5a9f6cd26eecba63`.
- `results/releases/pythia-2.8b-2026-08-11/`: GitHub Actions run `31466226471`, artifact `9097605754`, artifact digest `sha256:50f7e82eed8d23ad64aa431d27f9dd91e400718f863860118f016b249de5d9c5`, internal evidence TAR SHA-256 `a16909e4f67967b3c77a2bef215ef680a59d33229807ca36054b2160ed77840c`.
- `results/releases/tofu-llama32-1b-forget10-2026-08-11/`: GitHub Actions run `31490644488`, artifact `9104982056`, artifact digest `sha256:6a1381c76bec90233d5e4d1dde3fc43362405ed6bbd6d4378f709cdbf3634d8d`, internal evidence TAR SHA-256 `ac2fc36426c2b1fc91286c370956d48f061603ea1922ecabeda5f86faf832470`.

The TOFU release establishes exact single-GPU identity replay and exact trace-preserving forget10 replay from a physically redacted store on a second model family, Llama 3.2 1B Instruct. The frozen deletion request removes 400 of 4,000 TOFU records and preserves the benchmark answer-label masks. Standardized OpenUnlearning behavioral metrics remain a separate pending evaluation and are not inferred from the state-equality result.

None of the released results establishes distributed exactness, legal compliance, or superiority over approximate unlearning methods.
