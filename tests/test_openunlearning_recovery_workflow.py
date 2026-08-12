from pathlib import Path


def test_recovery_workflow_reuses_checkpoint_artifact_and_never_trains():
    workflow = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml').read_text()
    runner = Path('scripts/run_tofu_openunlearning_recovery_eval.sh').read_text()

    assert 'RECOVERY_ARTIFACT_ID: "9114652333"' in workflow
    assert 'RECOVERY_SOURCE_RUN_ID: "31519284938"' in workflow
    assert 'f7922732aacc4907d36191c59120e0bd6df2c0a38bc5d823744f2182ee1c6119' in workflow
    assert 'RECOVERY_ARTIFACT_SIZE: "9921106210"' in workflow
    assert 'actions: read' in workflow
    assert 'training_passes' in workflow and "'training_passes': 0" in workflow
    assert "'optimizer_updates': 0" in workflow
    assert 'reconstruct_tofu_for_openunlearning.py' not in workflow
    assert 'run_tofu_openunlearning_eval.sh' not in workflow
    assert 'run_tofu_openunlearning_recovery_eval.sh' in workflow
    assert 'MIN_FREE_GPU_MIB: "45000"' in workflow
    assert '--gpu-types "NVIDIA RTX A6000" "NVIDIA A40"' in workflow
    assert 'NVIDIA GeForce RTX 4090' not in workflow
    assert '"NVIDIA RTX A6000"|"NVIDIA A40") ;;' in workflow
    assert 'foreign_compute_processes_present=true' in workflow
    assert 'tofu-openunlearning-recovery.tar' in workflow
    assert 'sha256sum /workspace/recovery.zip' in workflow
    assert 'patch_openunlearning_bf16_numpy.py' in workflow

    assert 'reconstruct_tofu_for_openunlearning.py' not in runner
    assert 'optimizer.step' not in runner
    assert runner.count('openunlearning_adapter.py tofu-eval') == 2
    assert 'BF16 OpenUnlearning probability-metric execution gate passed' in runner


def test_recovery_workflow_retries_capacity_and_budget_rejections():
    workflow = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml').read_text()
    allocate = workflow.split('- name: Allocate one clean guarded evaluation Pod', 1)[1]
    allocate = allocate.split('- name: Download recovered canonical checkpoints and run evaluation only', 1)[0]

    assert 'for attempt in $(seq 1 "$MAX_GPU_ALLOCATION_ATTEMPTS")' in allocate
    assert 'if ! python scripts/runpod_control.py create' in allocate
    assert 'did not yield an acceptable <= $${MAX_HOURLY_COST}/h Pod' in allocate
    assert 'sleep 20' in allocate
    assert 'continue' in allocate
    assert '[[ "$accepted" == "1" ]]' in allocate


def test_recovery_workflow_uses_direct_hash_verified_json_handoff():
    workflow = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml').read_text()

    remote = workflow.split('- name: Download recovered canonical checkpoints and run evaluation only', 1)[1]
    remote = remote.split('- name: Collect and verify direct evaluation handoff', 1)[0]
    assert "Path('results/recovery-handoff')" in remote
    assert "'retain90_TOFU_EVAL.json'" in remote
    assert "'original_TOFU_EVAL.json'" in remote
    assert "'deletion_TOFU_EVAL.json'" in remote
    assert "'uas_evaluation_summary.json'" in remote
    assert "'reconstruction_summary.json'" in remote
    assert "'frozen_hashes.json'" in remote
    assert 'json.loads(source.read_text())' in remote
    assert 'hashlib.sha256(target.read_bytes()).hexdigest()' in remote
    assert "handoff / 'handoff-manifest.json'" in remote

    collect = workflow.split('- name: Collect and verify direct evaluation handoff', 1)[1]
    collect = collect.split('- name: Terminate RunPod Pod', 1)[0]
    assert 'id: collect' in collect
    assert 'for attempt in 1 2 3' in collect
    assert 'scp -r -i .runpod/id_ed25519' in collect
    assert '/workspace/unlearning/results/recovery-handoff' in collect
    assert '[[ "$copied" == "1" ]]' in collect
    assert "manifest.get('status') == 'passed'" in collect
    assert "json.loads(path.read_text())" in collect
    assert "observed == files[name]['sha256']" in collect
    assert "path.stat().st_size == files[name]['bytes']" in collect

    enforce = workflow.split('- name: Enforce evaluation and direct evidence-transfer success', 1)[1]
    assert 'steps.remote.outcome' in enforce
    assert 'steps.collect.outcome' in enforce
    assert 'gpu-artifacts/recovery-handoff/handoff-manifest.json' in enforce
    assert 'gpu-artifacts/recovery-handoff/original_TOFU_EVAL.json' in enforce
    assert 'gpu-artifacts/recovery-handoff/deletion_TOFU_EVAL.json' in enforce


def test_compact_archive_is_redundant_not_the_primary_handoff():
    workflow = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml').read_text()
    collect = workflow.split('- name: Collect and verify direct evaluation handoff', 1)[1]
    collect = collect.split('- name: Terminate RunPod Pod', 1)[0]

    assert 'Optional compact archive transfer attempt' in collect
    assert 'archive_copied=0' in collect
    assert 'direct hash-verified JSON handoff is complete and is the required evidence path' in collect
