# Reproducibility contract

Exactness in this project is scoped to a recorded execution environment. Deep-learning frameworks do not promise identical numerical behavior across arbitrary releases, devices, kernels, or distributed topologies, so environment capture and artifact locking are part of the experimental evidence.

## Required release artifacts

Every publication run should contain:

1. project Git commit SHA
2. `external/upstreams.lock.yaml`
3. `locks/artifact-sources.yaml`
4. frozen `locks/artifacts.lock.json`
5. resolved experiment configuration
6. full 40-character model commit SHA
7. full dataset commit SHA where a Hub dataset is used
8. prepared-dataset per-file SHA-256 values and directory digest
9. immutable execution plan and SHA-256
10. binary WAL and segment SHA-256
11. ordered sample-ID manifest and SHA-256
12. base and intermediate checkpoint hashes
13. model and optimizer state hashes
14. exactness comparison JSON
15. Python and package versions
16. PyTorch, CUDA and cuDNN versions
17. GPU model and memory
18. deterministic-algorithm settings
19. external benchmark command manifests when OpenUnlearning or MUSE is used

## Frozen artifact preflight

External research repositories are checked out detached at exact commits through `scripts/bootstrap_upstreams.py`.

After all prepared datasets exist, run:

```bash
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

The freeze step resolves readable Hugging Face refs to full commit SHAs and hashes every file in each prepared token store. The verification step fails if the source manifest changes, an external Git checkout moves, a Hub source is not represented by a full commit SHA, or prepared data bytes change.

The final `locks/artifacts.lock.json` is generated after data preparation and is included in the release bundle. It is not generated before the datasets exist because the file hashes would be meaningless.

## Release execution

Publication evidence should use:

```bash
python scripts/run_release.py <config>
python scripts/run_release_matrix.py configs/matrix-main.yaml
```

These runners verify the artifact lock and replace human-readable model refs such as Pythia `step143000` with the locked 40-character Hub commit before model loading. Direct `uas run` execution remains useful for development but is not the publication path.

## Deterministic defaults

The code sets a cuBLAS workspace configuration, disables cuDNN benchmarking, requests deterministic cuDNN behavior, disables TF32, requests deterministic PyTorch algorithms in strict mode, reseeds Python/NumPy/PyTorch RNGs per microbatch, stores the learning rate in the execution plan and preserves accumulation boundaries.

These controls define test conditions, not a theorem about every possible kernel.

## Training semantics

The exact target uses fixed microbatch slots, fixed sample ordering, fixed accumulation boundaries, sum-reduced token loss, a learning-rate value attached to each logical step, per-microbatch RNG seeds, an exact checkpoint before the first affected logical step and no optimizer update when an entire logical step contains zero retained loss.

The paper should use `trace-preserving counterfactual` whenever this distinction from repacked retraining matters.

## Benchmark supervision semantics

Generic WikiText stores do not contain `labels.npy`, so labels default to the input tokens and the system performs ordinary causal language modeling.

The first-class TOFU store contains `labels.npy`. `scripts/prepare_tofu_openunlearning.py` reproduces the pinned OpenUnlearning Llama 3.2 chat template and assigns `-100` to system/prompt/padding tokens, so only the final assistant response contributes training loss. Physical redaction preserves these labels.

## Physical redaction test

For main deletion runs, `materialize_redacted_store: true` writes a token store that physically omits requested rows before replay. The immutable execution plan may still reference those IDs because replay intercepts forgotten slots before any row lookup.

## External benchmark provenance

Publication TOFU and MUSE comparisons use the pinned OpenUnlearning checkout, not locally reimplemented baseline code. Every adapter invocation writes the upstream commit and exact commands into a machine-readable manifest. Exact replay/oracle states are exported to Hugging Face checkpoint format from their locked base revision with `scripts/export_hf_checkpoint.py` before upstream evaluation.

## CI versus scientific evidence

CI uses a tiny local causal model to detect software regressions. It is not scientific evidence for LLM-scale exactness. Paid GPU claims enter the paper only after the release lock, large-model runs and claims ledger are frozen.
