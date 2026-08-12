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
    assert '--gpu-types "NVIDIA RTX A6000"' in workflow
    assert 'tofu-openunlearning-recovery.tar' in workflow
    assert 'sha256sum /workspace/recovery.zip' in workflow
    assert 'patch_openunlearning_bf16_numpy.py' in workflow

    assert 'reconstruct_tofu_for_openunlearning.py' not in runner
    assert 'optimizer.step' not in runner
    assert runner.count('openunlearning_adapter.py tofu-eval') == 2
    assert 'BF16 OpenUnlearning probability-metric execution gate passed' in runner


def test_recovery_workflow_does_not_silently_drop_required_evidence():
    workflow = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml').read_text()

    collect = workflow.split('- name: Collect and verify compact evaluation evidence', 1)[1]
    collect = collect.split('- name: Terminate RunPod Pod', 1)[0]
    assert 'id: collect' in collect
    assert 'copied=0' in collect
    assert 'for attempt in 1 2 3' in collect
    assert '[[ "$copied" == "1" ]]' in collect
    assert 'sha256sum -c tofu-openunlearning-recovery-evidence.sha256' in collect
    assert 'evidence-manifest.txt' in collect
    assert 'checkpoint/binary payloads' in collect
    assert 'tofu-openunlearning-recovery-evidence.tar.gz" \\n                gpu-artifacts/tofu-openunlearning-recovery-evidence.tar.gz \\n              && scp' in collect
    assert 'tofu-openunlearning-recovery-evidence.tar.gz" \\n                gpu-artifacts/tofu-openunlearning-recovery-evidence.tar.gz || true' not in collect

    enforce = workflow.split('- name: Enforce evaluation and evidence-transfer success', 1)[1]
    assert 'steps.remote.outcome' in enforce
    assert 'steps.collect.outcome' in enforce
    assert 'test -s gpu-artifacts/evidence-manifest.txt' in enforce
