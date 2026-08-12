from pathlib import Path


WORKFLOW = Path('.github/workflows/runpod-tofu-openunlearning-recovery-eval.yml')
RUNNER = Path('scripts/run_tofu_openunlearning_recovery_eval.sh')
REMOTE_STEP = '- name: Download recovered canonical checkpoints, evaluate, and stream evidence'
COLLECT_STEP = '- name: Collect, verify, and freeze result bundle'
UPLOAD_STEP = '- name: Upload verified OpenUnlearning results before Pod termination'
TERMINATE_STEP = '- name: Terminate RunPod Pod'


def test_recovery_workflow_reuses_checkpoint_artifact_and_never_trains():
    workflow = WORKFLOW.read_text()
    runner = RUNNER.read_text()

    assert 'RECOVERY_ARTIFACT_ID: "9114652333"' in workflow
    assert 'RECOVERY_SOURCE_RUN_ID: "31519284938"' in workflow
    assert 'f7922732aacc4907d36191c59120e0bd6df2c0a38bc5d823744f2182ee1c6119' in workflow
    assert 'RECOVERY_ARTIFACT_SIZE: "9921106210"' in workflow
    assert 'actions: read' in workflow
    assert '"training_passes": 0' in workflow
    assert '"optimizer_updates": 0' in workflow
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
    assert 'zipfile.ZipFile(source)' in workflow
    assert 'patch_openunlearning_bf16_numpy.py' in workflow

    assert 'reconstruct_tofu_for_openunlearning.py' not in runner
    assert 'optimizer.step' not in runner
    assert runner.count('openunlearning_adapter.py tofu-eval') == 2
    assert 'BF16 OpenUnlearning probability-metric execution gate passed' in runner


def test_recovery_workflow_retries_capacity_and_budget_rejections():
    workflow = WORKFLOW.read_text()
    allocate = workflow.split('- name: Allocate one clean guarded evaluation Pod', 1)[1]
    allocate = allocate.split(REMOTE_STEP, 1)[0]

    assert 'for attempt in $(seq 1 "$MAX_GPU_ALLOCATION_ATTEMPTS")' in allocate
    assert 'if ! python scripts/runpod_control.py create' in allocate
    assert 'did not yield an acceptable <= $${MAX_HOURLY_COST}/h Pod' in allocate
    assert 'sleep 20' in allocate
    assert 'continue' in allocate
    assert '[[ "$accepted" == "1" ]]' in allocate


def test_remote_handoff_contains_hash_verified_openunlearning_outputs():
    workflow = WORKFLOW.read_text()
    remote = workflow.split(REMOTE_STEP, 1)[1]
    remote = remote.split(COLLECT_STEP, 1)[0]

    assert 'Path("/workspace/recovery-handoff")' in remote
    for name in (
        'retain90_TOFU_EVAL.json',
        'original_TOFU_EVAL.json',
        'original_uas_interop.json',
        'deletion_TOFU_EVAL.json',
        'deletion_uas_interop.json',
        'uas_evaluation_summary.json',
        'reconstruction_summary.json',
        'frozen_hashes.json',
        'bf16_numpy_patch.json',
        'gpu_probe.json',
        'runpod_allocation.json',
    ):
        assert name in remote
    assert 'json.loads(source.read_text())' in remote
    assert 'hashlib.sha256(target.read_bytes()).hexdigest()' in remote
    assert 'handoff / "handoff-manifest.json"' in remote
    assert 'tar -czf /workspace/recovery-handoff.tar.gz' in remote


def test_watchdog_cannot_hold_result_stream_open():
    workflow = WORKFLOW.read_text()
    remote = workflow.split(REMOTE_STEP, 1)[1]
    remote = remote.split(COLLECT_STEP, 1)[0]

    watchdog = 'nohup bash -lc "sleep ${SELF_DESTRUCT}; runpodctl pod delete \\$RUNPOD_POD_ID"'
    assert watchdog in remote
    assert '</dev/null >/tmp/uas-self-destruct.log 2>&1 &' in remote
    assert 'exec 3>&1' in remote
    assert 'exec 1>&2' in remote
    assert 'cat /workspace/recovery-handoff.tar.gz >&3' in remote
    assert 'exec 3>&-' in remote
    assert 'Remote evidence stream closed cleanly.' in remote

    # The self-destruct process must be spawned before fd 3 exists so it cannot
    # inherit the binary result channel and keep SSH alive after evaluation.
    assert remote.index(watchdog) < remote.index('exec 3>&1')
    assert remote.index('cat /workspace/recovery-handoff.tar.gz >&3') < remote.index('exec 3>&-')


def test_results_are_streamed_verified_and_uploaded_before_pod_termination():
    workflow = WORKFLOW.read_text()
    collect = workflow.split(COLLECT_STEP, 1)[1]
    collect = collect.split(UPLOAD_STEP, 1)[0]

    assert 'id: collect' in collect
    assert "steps.remote.outcome == 'success'" in collect
    assert 'test -s recovery-handoff.tar.gz' in collect
    assert 'gzip -t recovery-handoff.tar.gz' in collect
    assert 'preserved-openunlearning-results' in collect
    assert 'tar -xzf recovery-handoff.tar.gz --strip-components=1' in collect
    assert 'manifest.get("status") == "passed"' in collect
    assert 'hashlib.sha256(path.read_bytes()).hexdigest() == files[name]["sha256"]' in collect
    assert 'path.stat().st_size == files[name]["bytes"]' in collect
    assert 'tofu-openunlearning-results.tar.gz' in collect
    assert 'tofu-openunlearning-results.sha256' in collect
    assert 'sha256sum -c tofu-openunlearning-results.sha256' in collect
    assert 'member_count="$(grep -c' in collect
    assert 'test "$member_count" -eq 12' in collect
    assert 'Forbidden checkpoint/binary payload found in result bundle.' in collect
    assert 'echo "evidence_ready=true" >> "$GITHUB_OUTPUT"' in collect

    # Collection must validate the archive already streamed over the evaluation
    # SSH session. A post-evaluation SCP/reconnect would recreate the failure mode
    # that the hardened workflow was designed to eliminate.
    assert 'scp ' not in collect
    assert 'ssh ' not in collect
    assert '/workspace/recovery-handoff/.' not in collect
    assert 'for attempt in 1 2 3' not in collect

    upload_pos = workflow.index(UPLOAD_STEP)
    terminate_pos = workflow.index(TERMINATE_STEP)
    assert upload_pos < terminate_pos

    upload = workflow.split(UPLOAD_STEP, 1)[1]
    upload = upload.split(TERMINATE_STEP, 1)[0]
    assert 'id: upload_results' in upload
    assert "steps.collect.outputs.evidence_ready == 'true'" in upload
    assert 'name: tofu-openunlearning-results' in upload
    assert 'tofu-openunlearning-results.tar.gz' in upload
    assert 'tofu-openunlearning-results.sha256' in upload
    assert 'if-no-files-found: error' in upload


def test_final_gate_requires_verified_result_upload():
    workflow = WORKFLOW.read_text()
    enforce = workflow.split('- name: Enforce evaluation and preserved-result success', 1)[1]

    assert 'steps.remote.outcome' in enforce
    assert 'steps.collect.outcome' in enforce
    assert 'steps.collect.outputs.evidence_ready' in enforce
    assert 'steps.upload_results.outcome' in enforce
    assert 'verified OpenUnlearning result artifact was not uploaded before Pod termination' in enforce
