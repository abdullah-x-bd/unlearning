#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export UAS_REPO_ROOT="$REPO_ROOT"

FIRST_EVAL_MINUTES="${FIRST_EVAL_MINUTES:-75}"
SECOND_EVAL_MINUTES="${SECOND_EVAL_MINUTES:-45}"
RESULTS_DIR="$REPO_ROOT/results"
RECON_DIR="$REPO_ROOT/runs/tofu-openunlearning-reconstruction"
OPENUNLEARNING_DIR="$RESULTS_DIR/openunlearning/tofu"
FROZEN="$RESULTS_DIR/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json"
EVIDENCE_TAR="$RESULTS_DIR/tofu-openunlearning-recovery-evidence.tar.gz"
EVIDENCE_SHA="$RESULTS_DIR/tofu-openunlearning-recovery-evidence.sha256"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/tofu-openunlearning-recovery-eval.log") 2>&1

package_evidence() {
  df -h > "$RESULTS_DIR/tofu-openunlearning-recovery-disk-after.txt" 2>&1 || true
  python - <<'PY'
from pathlib import Path
import hashlib
import os
import tarfile

repo = Path(os.environ['UAS_REPO_ROOT']).resolve()
results = repo / 'results'
output = results / 'tofu-openunlearning-recovery-evidence.tar.gz'
allowed_suffixes = {'.json', '.jsonl', '.txt', '.log', '.yaml', '.yml', '.csv'}
forbidden_suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.npy', '.npz'}

files = []
openunlearning = results / 'openunlearning' / 'tofu'
if openunlearning.exists():
    for path in sorted(openunlearning.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(openunlearning)
        if 'checkpoint' in rel.parts or path.suffix.lower() in forbidden_suffixes:
            continue
        if path.suffix.lower() in allowed_suffixes:
            files.append(path)

singles = [
    results / 'runpod-allocation.json',
    results / 'tofu-openunlearning-recovery-eval.log',
    results / 'tofu-openunlearning-recovery-gpu-probe.json',
    results / 'tofu-openunlearning-recovery-disk-before.txt',
    results / 'tofu-openunlearning-recovery-disk-after.txt',
    results / 'openunlearning' / 'bf16-numpy-patch.json',
    repo / 'runs' / 'tofu-openunlearning-reconstruction' / 'summary.json',
    results / 'releases' / 'tofu-llama32-1b-forget10-2026-08-11' / 'frozen-hashes.json',
]
for path in singles:
    if path.exists() and path not in files:
        files.append(path)

with tarfile.open(output, 'w:gz') as archive:
    for path in sorted(files):
        archive.add(path, arcname=str(path.relative_to(repo)))

digest = hashlib.sha256(output.read_bytes()).hexdigest()
(results / 'tofu-openunlearning-recovery-evidence.sha256').write_text(
    f'{digest}  {output.name}\n'
)
print(f'OpenUnlearning recovery evidence files: {len(files)}', flush=True)
print(f'OpenUnlearning recovery evidence bytes: {output.stat().st_size}', flush=True)
print(f'OpenUnlearning recovery evidence TAR SHA256: {digest}', flush=True)
PY
}

verify_evidence() {
  test -s "$EVIDENCE_TAR"
  test -s "$EVIDENCE_SHA"
  (
    cd "$RESULTS_DIR"
    sha256sum -c "$(basename "$EVIDENCE_SHA")"
  )
  python - <<'PY'
from pathlib import Path
import os
import tarfile

repo = Path(os.environ['UAS_REPO_ROOT']).resolve()
archive = repo / 'results' / 'tofu-openunlearning-recovery-evidence.tar.gz'
required = {
    'results/openunlearning/tofu/reference/retain90/TOFU_EVAL.json',
    'results/openunlearning/tofu/models/uas_frozen_original_forget10/TOFU_EVAL.json',
    'results/openunlearning/tofu/models/uas_frozen_original_forget10/uas_interop.json',
    'results/openunlearning/tofu/models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json',
    'results/openunlearning/tofu/models/uas_frozen_trace_delete_forget10/uas_interop.json',
    'results/openunlearning/tofu/uas_evaluation_summary.json',
    'runs/tofu-openunlearning-reconstruction/summary.json',
    'results/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json',
}
forbidden_suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.npy', '.npz'}
with tarfile.open(archive, 'r:gz') as handle:
    names = set(handle.getnames())
missing = sorted(required - names)
assert not missing, f'missing required evidence members: {missing}'
for name in names:
    path = Path(name)
    assert 'checkpoint' not in path.parts, f'checkpoint leaked into compact evidence: {name}'
    assert path.suffix.lower() not in forbidden_suffixes, f'binary leaked into compact evidence: {name}'
print(f'Compact recovery evidence gate passed with {len(names)} members', flush=True)
PY
}

on_exit() {
  rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 ]]; then
    set +e
    package_evidence
  fi
  exit "$rc"
}
trap on_exit EXIT

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

df -h > "$RESULTS_DIR/tofu-openunlearning-recovery-disk-before.txt"
python scripts/gpu_probe.py --output "$RESULTS_DIR/tofu-openunlearning-recovery-gpu-probe.json"
python scripts/bootstrap_upstreams.py
test "$(git -C external/open-unlearning rev-parse HEAD)" = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"

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
for rel in ['hf/original/config.json', 'hf/deletion/config.json']:
    path = Path('runs/tofu-openunlearning-reconstruction') / rel
    assert path.exists(), path
print('Recovered canonical checkpoint summary gate passed; no reconstruction required', flush=True)
PY

python -m pip install --no-cache-dir -e 'external/open-unlearning[lm-eval]'
python -m pip install --no-cache-dir 'lm-eval[hf]==0.4.11'
python scripts/patch_openunlearning_bf16_numpy.py

python - <<'PY'
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path('external/open-unlearning/src').resolve()))
from evals.metrics.utils import evaluate_probability

assert torch.cuda.is_available()
device = torch.device('cuda')
class Dummy:
    device = device
    def __call__(self, **batch):
        class Out: pass
        out = Out()
        out.logits = torch.tensor(
            [[[1.,2.,3.,4.],[4.,3.,2.,1.],[1.,1.,1.,1.]]],
            dtype=torch.bfloat16,
            device=device,
        )
        return out

batch = {
    'input_ids': torch.tensor([[1,2,3]], device=device),
    'attention_mask': torch.tensor([[1,1,1]], device=device),
    'labels': torch.tensor([[-100,2,3]], device=device),
}
result = evaluate_probability(Dummy(), batch)
assert len(result) == 1 and isinstance(result[0]['avg_loss'], float)
assert isinstance(result[0]['prob'], float)
print('BF16 OpenUnlearning probability-metric execution gate passed', flush=True)
PY

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
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_original_forget10/uas_interop.json"
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json"
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
    'reused_hash_verified_recovery_checkpoints': True,
    'bf16_numpy_interop_patch': str(root.parent / 'bf16-numpy-patch.json'),
    'reference_log': str(root / 'reference/retain90/TOFU_EVAL.json'),
    'original_log': str(root / 'models/uas_frozen_original_forget10/TOFU_EVAL.json'),
    'deletion_log': str(root / 'models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json'),
}
(root / 'uas_evaluation_summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY

package_evidence
verify_evidence
trap - EXIT

echo "Recovered canonical checkpoints evaluated successfully; compact evidence verified; no optimizer updates were run."
