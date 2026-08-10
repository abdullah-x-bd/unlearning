# Reproducibility contract

Exactness in this project is scoped to a recorded execution environment. PyTorch does not guarantee identical results across releases, platforms, or CPU/GPU execution, so every scientific run must preserve enough environment information to reproduce the stack.

## Required run artifacts

Every released run should contain:

1. resolved experiment configuration
2. immutable execution plan and its SHA-256
3. binary WAL and segment SHA-256
4. ordered sample-ID manifest
5. prepared-dataset manifest and content hashes
6. base-model name and immutable revision or commit
7. checkpoint hashes
8. model and optimizer state hashes
9. `environment.json`
10. `pip freeze`
11. GPU name, CUDA version, cuDNN version, and PyTorch version
12. exactness comparison JSON

## Deterministic defaults

The code enables deterministic PyTorch algorithms, disables cuDNN benchmarking, disables TF32, sets a cuBLAS workspace configuration, and reseeds each microbatch from the execution plan.

These settings improve reproducibility but are not treated as a theorem about every kernel. The experiment reports any deterministic-operation exception or state mismatch.

## Training semantics

The exact target uses:

- fixed microbatch slots
- fixed accumulation boundaries
- sum-reduced token loss
- a learning-rate value attached to each logical step
- independent per-microbatch RNG seeds
- an exact checkpoint from before the first affected logical step
- no optimizer update when an entire logical step has zero retained loss

The release paper must use the phrase `trace-preserving counterfactual` whenever this distinction matters.

## Dataset immutability

Do not depend on a floating remote dataset revision for a released result. Prepare the dataset once, record the upstream revision when available, and publish the preparation manifest plus cryptographic hashes. Large token arrays may be attached to a release or archived separately.

## Model immutability

Replace `revision: main` with a model commit SHA for release runs.

## CI

CI uses a tiny local causal model only to detect software regressions in WAL serialization, checkpoint replay, and XOR rollback. CI results are not paper results.
