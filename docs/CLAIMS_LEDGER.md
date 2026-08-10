# Claims ledger

No claim moves into the rewritten paper until its evidence cell is populated by a released run artifact.

| ID | Candidate claim | Required evidence | Status |
| --- | --- | --- | --- |
| G1 | WAL reconstruction reproduces the execution plan | plan/WAL equality across all main runs | pending large runs |
| G2 | trace-preserving replay can be byte exact | exact model and optimizer hashes across model scales and seeds | pending large runs |
| G3 | slot masking is more robust than physical filtering | determinism stress matrix | pending |
| G4 | repacked retraining is generally not byte equivalent | repacked baseline comparisons | pending |
| G5 | checkpoint spacing trades storage for replay latency | checkpoint sweep with measured bytes and time | pending |
| G6 | XOR patches exactly restore recent states | unit tests plus scaled storage/latency sweep | core test implemented |
| G7 | cohort adapter deletion exactly recovers the frozen base | LoRA cohort experiment and base hash equality | implementation scaffolded |
| G8 | curvature hot path reduces forget evidence before exact replay | forget/retain audits versus baselines | implementation scaffolded |
| G9 | distributed replay is exact | multi-GPU deterministic suite | not yet eligible for claim |
| G10 | the mechanism operationalizes legal erasure requirements | legal analysis distinct from technical results | not a technical claim |
