from pathlib import Path


def test_bf16_numpy_patch_is_minimal_and_recovery_runner_has_no_training():
    patcher = Path('scripts/patch_openunlearning_bf16_numpy.py').read_text()
    runner = Path('scripts/run_tofu_openunlearning_recovery_eval.sh').read_text()

    assert 'avg_losses.float().cpu().numpy().tolist()' in patcher
    assert 'normalized_probs.float().cpu().numpy().tolist()' in patcher
    assert 'expected exactly one pinned occurrence' in patcher
    assert '4ad738aaf60f6a4385f6e2506d01da99e76c31f3' in patcher

    assert 'reconstruct_tofu_for_openunlearning.py' not in runner
    assert '625' not in runner
    assert 'optimizer updates' not in runner.lower().replace('no optimizer updates', '')
    assert runner.count('openunlearning_adapter.py tofu-eval') == 2
    assert 'Recovered canonical checkpoint summary gate passed' in runner
    assert 'BF16 OpenUnlearning probability-metric execution gate passed' in runner
    assert 'patch_openunlearning_bf16_numpy.py' in runner
    assert 'reused_hash_verified_recovery_checkpoints' in runner
