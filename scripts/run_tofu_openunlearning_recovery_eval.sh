#!/usr/bin/env bash
set -Eeuo pipefail

FIRST_EVAL_MINUTES="${FIRST_EVAL_MINUTES:-75}"
SECOND_EVAL_MINUTES="${SECOND_EVAL_MINUTES:-45}"
RESULTS_DIR="results"
RECON_DIR="runs/tofu-openunlearning-reconstruction"
OPENUNLEARNING_DIR="$RESULTS_DIR/openunlearning/tofu"
FROZEN="$RESULTS_DIR/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/tofu-openunlearning-recovery-eval.log") 2>&1

package_evidence() {
  set +e
  df -h > "$RESULTS_DIR/tofu-openunlearning-recovery-disk-after.txt" 2>&1
  python - <<'PY'
from pathlib import Path
import hashlib
import tarfile

output = Path('results/tofu-openunlearning-recovery-evidence.tar.gz')
roots = [Path('results/openunlearning/tofu')]
singles = [
    Path('results/runpod-allocation.json'),
    Path('results/tofu-openunlearning-recovery-eval.log'),
    Path('results/tofu-openunlearning-recovery-gpu-probe.json'),
    Path('results/tofu-openunlearning-recovery-disk-before.txt'),
    Path('results/tofu-openunlearning-recovery-disk-after.txt'),
    Path('results/openunlearning/bf16-numpy-patch.json'),
    Path('runs/tofu-openunlearning-reconstruction/summary.json'),
    Path('results/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json'),
]
with tarfile.open(output, 'w:gz') as archive:
    for root in roots:
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.add(path, arcname=str(path))
    for path in singles:
        if path.exists():
            archive.add(path, arcname=str(path))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
Path('results/tofu-openunlearning-recovery-evidence.sha256').write_text(
    f'{digest}  {output.name}\n'
)
print(f'OpenUnlearning recovery evidence TAR SHA256: {digest}', flush=True)
PY
}
trap package_evidence EXIT

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

class Dummy:
    device = torch.device('cpu')
    def __call__(self, **batch):
        class Out: pass
        out = Out()
        # Shape [batch=1, seq=3, vocab=4], BF16 specifically exercises the fixed path.
        out.logits = torch.tensor([[[1.,2.,3.,4.],[4.,3.,2.,1.],[1.,1.,1.,1.]]], dtype=torch.bfloat16)
        return out

batch = {
    'input_ids': torch.tensor([[1,2,3]]),
    'attention_mask': torch.tensor([[1,1,1]]),
    'labels': torch.tensor([[-100,2,3]]),
}
result = evaluate_probability(Dummy(), batch)
assert len(result) == 1 and isinstance(result[0]['avg_loss'], float)
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
test -f "$OPENUNLEARNING_DIR/models/uas_frozen_trace_delete_forget10/TOFU_EVAL.json"

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

echo "Recovered canonical checkpoints evaluated successfully; no optimizer updates were run."
