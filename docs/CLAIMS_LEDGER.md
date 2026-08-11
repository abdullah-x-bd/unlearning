# Claims ledger

No candidate claim moves into the rebuilt manuscript until its required evidence is present in a released artifact.

| ID | Candidate claim | Required evidence | Current status |
| --- | --- | --- | --- |
| G1 | WAL reconstruction reproduces the intended execution plan | plan/WAL equality across release runs | implemented, GPU evidence pending |
| G2 | no-deletion replay can reproduce the original run | model and optimizer identity replay across release matrix | implementation complete, runs pending |
| G3 | trace-preserving deletion replay can be byte exact | oracle versus replay hashes and tensor comparisons across scales and seeds | runs pending |
| G4 | physical deletion of token rows is compatible with replay | replay against materialized redacted stores | implementation complete, scale runs pending |
| G5 | slot masking is more robust than physical filtering | determinism and deletion matrix | runs pending |
| G6 | repacked retain-only retraining can diverge from the trace-preserving target | same-checkpoint repacked comparisons | runs pending |
| G7 | specific provenance fields are empirically necessary | one-field provenance ablations | implementation complete, runs pending |
| G8 | checkpoint spacing creates a storage versus replay-latency tradeoff | checkpoint interval sweep | implementation complete, runs pending |
| G9 | exact and near-duplicate closure changes deletion scope | closure counts and downstream experiments | implementation complete, runs pending |
| G10 | XOR rollback restores earlier state bytes exactly | unit tests plus scaled checkpoint benchmark | regression test passes, scale benchmark pending |
| G11 | cohort adapter deletion can recover the unchanged base exactly | base hash before adapter and after unload | implementation complete, GPU run pending |
| G12 | curvature hot path provides useful approximate deletion behavior | forget/retain/audit comparison against baselines | implementation complete, runs pending |
| G13 | replay compares with established approximate methods on standardized unlearning metrics | same-target GradAscent, GradDiff, NPO and SimNPO runs from pinned OpenUnlearning plus exact command manifests | integration complete, runs pending |
| G14 | the mechanism generalizes beyond the controlled WikiText systems study | Llama 3.2 1B TOFU trace runs plus pinned OpenUnlearning evaluation on forget01/05/10 | integration complete, runs pending |
| G15 | results are reproducible from frozen external dependencies and data | verified artifact lock containing full Hub SHAs, external Git commits and prepared-dataset file hashes | lock machinery complete, final data lock pending |
| G16 | MUSE confirms behavior under larger external validation including scaling/sequential deletion | pinned MUSE News/Books results | budget-dependent, no claim yet |
| G17 | distributed replay is exact | dedicated multi-GPU deterministic suite | not implemented, claim prohibited |
| G18 | the mechanism directly satisfies a specific legal erasure obligation | separate legal analysis tied to actual technical guarantees | not a technical claim |
