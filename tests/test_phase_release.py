import torch

from unlearning_at_scale.lifecycle import release_phase


def test_release_invalidates_caller_owned_model_and_optimizer_storage():
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 8)).sum()
    loss.backward()
    optimizer.step()

    assert optimizer.state
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())

    release_phase(optimizer, model)

    assert not optimizer.state
    assert all(not group["params"] for group in optimizer.param_groups)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
