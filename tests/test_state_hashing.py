from pathlib import Path

import torch

from unlearning_at_scale.state import optimizer_sha256, save_checkpoint, state_sha256


def test_state_sha256_handles_zero_dimensional_and_bfloat16_tensors():
    scalar = torch.tensor(3.0, dtype=torch.float32)
    scalar_changed = torch.tensor(4.0, dtype=torch.float32)
    bf16 = torch.tensor(1.5, dtype=torch.bfloat16)

    first = state_sha256({"scalar": scalar, "bf16": bf16})
    second = state_sha256({"scalar": scalar.clone(), "bf16": bf16.clone()})
    changed = state_sha256({"scalar": scalar_changed, "bf16": bf16})

    assert len(first) == 64
    assert first == second
    assert first != changed


def test_adamw_optimizer_state_and_checkpoint_hash_after_update(tmp_path: Path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.ones((3, 4))
    loss = model(inputs).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    scalar_states = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.ndim == 0
    ]
    assert scalar_states

    optimizer_hash = optimizer_sha256(optimizer)
    metadata = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        next_optimizer_step=1,
    )

    assert len(optimizer_hash) == 64
    assert metadata["optimizer_sha256"] == optimizer_hash
    assert len(metadata["model_sha256"]) == 64
    assert metadata["bytes"] > 0
