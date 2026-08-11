#!/usr/bin/env bash
set -Eeuo pipefail

RECONSTRUCTION_MINUTES="${RECONSTRUCTION_MINUTES:-90}"
FIRST_EVAL_MINUTES="${FIRST_EVAL_MINUTES:-75}"
SECOND_EVAL_MINUTES="${SECOND_EVAL_MINUTES:-45}"
RESULTS_DIR="results"
RECON_DIR="runs/tofu-openunlearning-reconstruction"
OPENUNLEARNING_DIR="$RESULTS_DIR/openunlearning/tofu"
FROZEN="$RESULTS_DIR/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/tofu-openunlearning-eval.log") 2>&1

package_evidence() {
  set +e
  df -h > "$RESULTS_DIR/tofu-openunlearning-disk-after.txt" 2>&1
  python - <<'PY'
from pathlib import Path
import hashlib
import tarfile

output = Path('results/tofu-openunlearning-evidence.tar.gz')
roots = [
    Path('results/openunlearning/tofu'),
    Path('runs/tofu-openunlearning-reconstruction'),
]
singles = [
    Path('results/runpod-allocation.json'),
    Path('results/tofu-openunlearning-eval.log'),
    Path('results/tofu-openunlearning-gpu-probe.json'),
    Path('results/tofu-openunlearning-phase-release-smoke.json'),
    Path('results/tofu-openunlearning-disk-before.txt'),
    Path('results/tofu-openunlearning-disk-after.txt'),
    Path('results/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json'),
]
excluded_suffixes = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
excluded_parts = {'redacted-data', '.cache'}
with tarfile.open(output, 'w:gz') as archive:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix in excluded_suffixes:
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            archive.add(path, arcname=str(path))
    for path in singles:
        if path.exists():
            archive.add(path, arcname=str(path))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
Path('results/tofu-openunlearning-evidence.sha256').write_text(
    f'{digest}  {output.name}\n'
)
print(f'OpenUnlearning evidence TAR SHA256: {digest}', flush=True)
PY
}
trap package_evidence EXIT

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=2026
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled

df -h > "$RESULTS_DIR/tofu-openunlearning-disk-before.txt"
python scripts/gpu_probe.py --output "$RESULTS_DIR/tofu-openunlearning-gpu-probe.json"
python scripts/bootstrap_upstreams.py
python scripts/prepare_release_data.py --clean --wikitext-max-records 20000
python scripts/validate_release_data.py
python scripts/verify_release_lock.py

python scripts/gpu_phase_release_smoke.py \
  --config configs/benchmarks/tofu-llama32-1b-forget10.yaml \
  --output "$RESULTS_DIR/tofu-openunlearning-phase-release-smoke.json"

readarray -t HASHES < <(python - "$FROZEN" <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))
for key in [
    'plan_sha256',
    'forget_ids_sha256',
    'original_model_sha256',
    'original_optimizer_sha256',
    'deletion_model_sha256',
    'deletion_optimizer_sha256',
]:
    print(f[key])
PY
)

timeout --signal=TERM --kill-after=180s "${RECONSTRUCTION_MINUTES}m" \
  python scripts/reconstruct_tofu_for_openunlearning.py \
    --expected-plan-sha256 "${HASHES[0]}" \
    --expected-forget-sha256 "${HASHES[1]}" \
    --expected-original-model-sha256 "${HASHES[2]}" \
    --expected-original-optimizer-sha256 "${HASHES[3]}" \
    --expected-deletion-model-sha256 "${HASHES[4]}" \
    --expected-deletion-optimizer-sha256 "${HASHES[5]}"

python - <<'PY'
import json
from pathlib import Path
frozen = json.loads(Path('results/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json').read_text())
rebuilt = json.loads(Path('runs/tofu-openunlearning-reconstruction/summary.json').read_text())
assert rebuilt['status'] == 'passed'
assert rebuilt['original']['model_sha256'] == frozen['original_model_sha256']
assert rebuilt['original']['optimizer_sha256'] == frozen['original_optimizer_sha256']
assert rebuilt['deletion']['model_sha256'] == frozen['deletion_model_sha256']
assert rebuilt['deletion']['optimizer_sha256'] == frozen['deletion_optimizer_sha256']
assert rebuilt['plan_sha256'] == frozen['plan_sha256']
assert rebuilt['forget_ids_sha256'] == frozen['forget_ids_sha256']
print('Canonical state hash gate passed; standardized evaluation may proceed', flush=True)
PY

# Keep the canonical two-pass reconstruction in the project-pinned runtime.
# Only after its hashes match do we install the pinned external evaluator.
python -m pip install --no-cache-dir -e external/open-unlearning
test "$(git -C external/open-unlearning rev-parse HEAD)" = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"

timeout --signal=TERM --kill-after=120s "${FIRST_EVAL_MINUTES}m" \
  python scripts/openunlearning_adapter.py tofu-eval \
    --checkpoint "$RECON_DIR/hf/original" \
    --forget-split forget10 \
    --task-name uas_frozen_original_forget10 \
    --attention-implementation eager \
    --output-root "$OPENUNLEARNING_DIR"

timeout --signal=TERM --kill-after=120s "${SECOND_EVAL_MINUTES}m" \
  python scripts/openunlearning_adapter.py tofu-eval \
    --checkpoint "$RECON_DIR/hf/deletion" \
    --forget-split forget10 \
    --task-name uas_frozen_trace_delete_forget10 \
    --attention-implementation eager \
    --output-root "$OPENUNLEARNING_DIR"

test -f "$OPENUNLEARNING_DIR/reference/retain90/TOFU_EVAL.json"
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_original_forget10/TOFU_EVAL.json"
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json"
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_original_forget10/uas_interop.json"
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_trace_delete_forget10/uas_interop.json"

python - <<'PY'
import json
from pathlib import Path
root = Path('results/openunlearning/tofu')
payload = {
    'status': 'passed',
    'framework': 'OpenUnlearning',
    'framework_commit': '4ad738aaf60f6a4385f6e2506d01da99e76c31f3',
    'benchmark': 'TOFU',
    'forget_split': 'forget10',
    'holdout_split': 'holdout10',
    'retain_split': 'retain90',
    'attention_implementation': 'eager',
    'reference_log': str(root / 'reference/retain90/TOFU_EVAL.json'),
    'original_log': str(root / 'models/uas_frozen_original_forget10/TOFU_EVAL.json'),
    'deletion_log': str(root / 'models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json'),
}
(root / 'uas_evaluation_summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY

echo "Canonical TOFU reconstruction and pinned OpenUnlearning evaluation completed successfully."
