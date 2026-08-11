#!/usr/bin/env bash
set -Eeuo pipefail

MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-300}"
RESULTS_DIR="results"
RUN_DIR="runs/pythia-160m"
mkdir -p "$RESULTS_DIR"

exec > >(tee -a "$RESULTS_DIR/gpu-run.log") 2>&1

package_evidence() {
  python scripts/package_gpu_evidence.py \
    --run-dir "$RUN_DIR" \
    --output "$RESULTS_DIR/pythia-160m-evidence.tar.gz" \
    --config configs/pythia-160m.yaml \
    --results-dir "$RESULTS_DIR" \
    --locks-dir locks || true
}
trap package_evidence EXIT

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=2026
export CUBLAS_WORKSPACE_CONFIG=:4096:8

printf '%s\n' "$(git rev-parse HEAD)" > "$RESULTS_DIR/gpu-repository-commit.txt"
python scripts/gpu_probe.py --output "$RESULTS_DIR/gpu-probe.json"
nvidia-smi -q > "$RESULTS_DIR/gpu-nvidia-smi.txt"
python -m pip freeze > "$RESULTS_DIR/gpu-pip-freeze.txt"

python scripts/bootstrap_upstreams.py
python scripts/prepare_release_data.py --clean --wikitext-max-records 20000
python scripts/validate_release_data.py
python scripts/verify_release_lock.py

python - <<'PY'
import json
from pathlib import Path
from unlearning_at_scale.determinism import environment_snapshot
Path("results/gpu-environment-before-run.json").write_text(
    json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n"
)
PY

set +e
timeout --signal=TERM --kill-after=120s "${MAX_RUNTIME_MINUTES}m" \
  python scripts/run_release.py configs/pythia-160m.yaml
status=$?
set -e
printf '%s\n' "$status" > "$RESULTS_DIR/gpu-exit-code.txt"

python - <<'PY'
import json
from pathlib import Path
from unlearning_at_scale.determinism import environment_snapshot
Path("results/gpu-environment-after-run.json").write_text(
    json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n"
)
PY

if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
  echo "GPU release exceeded MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES}" >&2
  exit "$status"
fi
if [[ "$status" -ne 0 ]]; then
  echo "GPU release failed with exit code $status" >&2
  exit "$status"
fi

echo "Pythia 160M release experiment completed successfully."
