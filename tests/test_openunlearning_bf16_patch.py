from pathlib import Path


def test_openunlearning_bf16_patch_targets_probability_outputs_only():
    source = Path('scripts/patch_openunlearning_bf16_numpy.py').read_text()
    assert source.count('avg_losses = avg_losses.float().cpu().numpy().tolist()') == 1
    assert source.count('normalized_probs = normalized_probs.float().cpu().numpy().tolist()') == 1
    assert 'Represent BF16 scalar metric tensors as float32 before NumPy conversion' in source
