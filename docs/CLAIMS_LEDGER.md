# Claims ledger

No candidate claim moves into the rebuilt manuscript until its required evidence is present in a released artifact.

| ID | Candidate claim | Required evidence | Current status |
| --- | --- | --- | --- |
| G1 | WAL reconstruction reproduces the intended execution plan | plan/WAL equality across release runs | supported at Pythia 160M and Pythia 2.8B; both release runs passed the WAL-to-plan equality guard |
| G2 | no-deletion replay can reproduce the original run | model and optimizer identity replay across release matrix | exact model and optimizer identity at Pythia 160M and Pythia 2.8B; additional seeds remain untested |
| G3 | trace-preserving deletion replay can be byte exact | oracle versus replay hashes and tensor comparisons across scales and seeds | exact at Pythia 160M for early 0.1%, middle 1%, late 1% and random 5%; exact again at Pythia 2.8B for random 5%; additional seed replication pending |
| G4 | physical deletion of token rows is compatible with replay | replay against materialized redacted stores | supported at Pythia 160M across four scenarios and at Pythia 2.8B for random 5%; forgotten IDs were absent from each tested materialized replay store |
| G5 | slot masking is more robust than physical filtering | determinism and deletion matrix | Pythia 160M release shows exact slot-mask replay and non-exact filter replay in all four scenarios; filter was intentionally not repeated at 2.8B |
| G6 | repacked retain-only retraining can diverge from the trace-preserving target | same-checkpoint repacked comparisons | supported at Pythia 160M in all four release scenarios; repacked retraining was intentionally not repeated at 2.8B |
| G7 | specific provenance fields are empirically necessary | one-field provenance ablations | implementation complete, runs pending |
| G8 | checkpoint spacing creates a storage versus replay-latency tradeoff | checkpoint interval sweep | implementation complete, runs pending |
| G9 | exact and near-duplicate closure changes deletion scope | closure counts and downstream experiments | implementation complete, runs pending |
| G10 | XOR rollback restores earlier state bytes exactly | unit tests plus scaled checkpoint benchmark | regression test passes, scale benchmark pending |
| G11 | cohort adapter deletion can recover the unchanged base exactly | base hash before adapter and after unload | implementation complete, GPU run pending |
| G12 | curvature hot path provides useful approximate deletion behavior | forget/retain/audit comparison against baselines | implementation complete, runs pending |
| G13 | replay compares with established approximate methods on standardized unlearning metrics | same-target GradAscent, GradDiff, NPO and SimNPO runs from pinned OpenUnlearning plus exact command manifests | integration complete, runs pending |
| G14 | the mechanism generalizes beyond the controlled WikiText systems study | Llama 3.2 1B TOFU trace runs plus pinned OpenUnlearning evaluation on forget01/05/10 | integration complete, runs pending |
| G15 | results are reproducible from frozen external dependencies and data | verified artifact lock containing full Hub SHAs, external Git commits and prepared-dataset file hashes | frozen artifact lock verified; CPU dataset preparation reproduced byte-for-byte across clean runners; successful Pythia 160M and 2.8B GPU releases both executed behind the verified release-lock gate |
| G16 | MUSE confirms behavior under larger external validation including scaling/sequential deletion | pinned MUSE News/Books results | budget-dependent, no claim yet |
| G17 | distributed replay is exact | dedicated multi-GPU deterministic suite | not implemented, claim prohibited |
| G18 | the mechanism directly satisfies a specific legal erasure obligation | separate legal analysis tied to actual technical guarantees | not a technical claim |

Released empirical evidence:

- `results/releases/pythia-160m-2026-08-11/`: GitHub Actions artifact `9088115288`, digest `sha256:f643f9baea2e819ac2494b605d11ee3dc15d991d9e1bd9bc5a9f6cd26eecba63`.
- `results/releases/pythia-2.8b-2026-08-11/`: GitHub Actions run `31466226471`, artifact `9097605754`, artifact digest `sha256:50f7e82eed8d23ad64aa431d27f9dd91e400718f863860118f016b249de5d9c5`, internal evidence TAR SHA-256 `a16909e4f67967b3c77a2bef215ef680a59d33229807ca36054b2160ed77840c`.

The 2.8B release establishes exact single-GPU identity replay and exact random-5% trace-preserving deletion replay in the pinned A100 environment. It does not establish distributed exactness, standardized semantic unlearning quality, approximate-baseline superiority, or legal compliance.
