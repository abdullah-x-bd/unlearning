# Reproducibility contract

Exactness in this project is scoped to a recorded execution environment. Deep-learning frameworks do not promise identical numerical behavior across arbitrary releases, devices, kernels, or distributed topologies, so environment capture is part of the experimental evidence.

## Required release artifacts

Every release run should contain:

1. resolved experiment configuration
2. Git commit SHA when available
3. immutable model revision
4. resolved Hugging Face model commit when available
5. dataset manifest and hashes
6. immutable execution plan and SHA-256
7. binary WAL and segment SHA-256
8. ordered sample-ID manifest and SHA-256
9. base and intermediate checkpoint hashes
10. model and optimizer state hashes
11. exactness comparison JSON
12. Python version
13. package versions
14. PyTorch version
15. CUDA version
16. cuDNN version
17. GPU model and memory
18. deterministic-algorithm settings

## Release mode

Development configs may use a floating model revision. Release configs must not.

Setting:

```yaml
release_mode: true
```

causes the main experiment runner to reject `revision: main`, `revision: master`, or a missing revision.

## Deterministic defaults

The code:

- sets a cuBLAS workspace configuration
- disables cuDNN benchmarking
- requests deterministic cuDNN behavior
- disables TF32
- requests deterministic PyTorch algorithms in strict mode
- reseeds Python, NumPy, PyTorch CPU, and CUDA RNGs per microbatch
- stores the learning rate in the execution plan
- preserves accumulation boundaries

These controls are test conditions, not a theorem about every possible kernel.

## Training semantics

The exact target uses:

- fixed microbatch slots
- fixed sample ordering
- fixed accumulation boundaries
- sum-reduced token loss
- a learning-rate value attached to each logical step
- per-microbatch RNG seeds
- an exact checkpoint before the first affected logical step
- no optimizer update when a complete logical step contains zero retained loss

The paper should use `trace-preserving counterfactual` whenever this distinction from repacked retraining matters.

## Physical redaction test

For main deletion runs, enable:

```yaml
materialize_redacted_store: true
```

The experiment then writes a token store that physically omits the forgotten rows before replay. The original plan can still refer to those IDs because the replay policy intercepts them before row lookup.

## Dataset immutability

Do not regenerate a release dataset under an existing artifact name. Record the upstream dataset revision where available and archive the preparation manifest.

The dataset preparation scripts store cryptographic file hashes. WikiText preparation also stores content hashes and SimHash signatures used by duplicate-closure experiments.

## Model immutability

Replace floating `revision: main` values with model commit SHAs for release runs.

## CI versus scientific evidence

CI uses a tiny local causal model to detect regressions in WAL serialization, plan reconstruction, exact replay, physical redaction, duplicate closure, baseline objectives, and XOR rollback. CI is not scientific evidence for LLM-scale exactness.
