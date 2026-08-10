import torch

from unlearning_at_scale.rollback import apply_exact_patch, build_exact_patch


def test_xor_patch_restores_exact_tensor_bytes():
    before = {"model": {"w": torch.tensor([1.0, -2.5], dtype=torch.float32)}, "step": 4}
    after = {"model": {"w": torch.tensor([1.125, -2.0], dtype=torch.float32)}, "step": 5}
    patch = build_exact_patch(before, after)
    restored = apply_exact_patch(after, patch)
    assert torch.equal(restored["model"]["w"], before["model"]["w"])
    assert restored["step"] == 4
