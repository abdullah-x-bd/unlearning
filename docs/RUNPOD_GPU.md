# Guarded RunPod GPU execution

The first paid experiment is the frozen Pythia 160M release run. It is deliberately automated so GPU billing starts only after all CPU/reproducibility gates have already passed.

## One-shot trigger

`.github/workflows/runpod-pythia-160m.yml` supports manual `workflow_dispatch` and one automatic trigger when `gpu/trigger-pythia-160m-v1` first lands on `main`. Normal repository pushes do not rent GPUs.

## Allocation policy

The controller requests exactly one non-interruptible GPU and tries these types in order:

1. NVIDIA A40
2. NVIDIA RTX A6000
3. NVIDIA GeForce RTX 4090

Secure Cloud is attempted before Community Cloud. Allocation is rejected and immediately terminated if RunPod reports an effective hourly price above USD 0.75/hour.

The container image is immutable by Docker manifest digest:

`runpod/pytorch@sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`

The image corresponds to RunPod PyTorch 2.4.0, Python 3.11, CUDA 12.4.1, Ubuntu 22.04. The repository then installs the frozen research dependencies from `pyproject.toml`, including PyTorch 2.4.1 and Transformers 4.51.3.

## Budget guards

- one GPU only
- maximum reported GPU price: USD 0.75/hour
- experiment runtime limit: 300 minutes
- GitHub job limit: 360 minutes
- in-Pod self-destruction timer: 330 minutes
- Pod termination runs in an `always()` cleanup step
- no spot/interruptible Pod for the canonical exactness run

At the hourly cap, the 330-minute self-destruction ceiling bounds GPU rental at approximately USD 4.13 before separately billed storage/platform adjustments. This is intentionally below the initial USD 10 experiment budget.

## Remote preflight

Before training, the Pod must:

1. pass CUDA availability
2. expose exactly one GPU
3. support BF16
4. pass a same-process repeated CUDA matmul equality probe
5. install the frozen software stack
6. fetch the three pinned upstream research repositories
7. regenerate the frozen WikiText and TOFU token stores
8. reproduce the committed data validation checks
9. verify `locks/artifacts.lock.json`
10. record `nvidia-smi`, PyTorch/CUDA/cuDNN and package versions

Only then does `scripts/run_release.py configs/pythia-160m.yaml` begin.

## Evidence preservation

Large `.pt` checkpoints and token arrays stay off the GitHub Actions artifact. The compact evidence bundle contains all JSON summaries, execution plans, WAL files, ordered-ID manifests, environment records, logs, configs and artifact locks, plus an `EVIDENCE_MANIFEST.json` containing SHA-256 and byte size for every included file.

Evidence is copied from the Pod before termination and uploaded as the `pythia-160m-gpu-evidence` Actions artifact.

The canonical result is considered successful only if the remote release command exits zero. A failed run still attempts to preserve all evidence that existed at failure time, then terminates the Pod.
