import torch

from unlearning_at_scale.dataset import TokenBatch


def test_token_batch_to_moves_every_tensor_and_preserves_metadata():
    batch = TokenBatch(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        labels=torch.tensor([[-100, 2, 3]], dtype=torch.long),
        sample_weights=torch.tensor([1.0], dtype=torch.float32),
        sample_ids=("sample-1",),
        retained_count=1,
    )

    moved = batch.to("meta")

    assert moved.input_ids.device.type == "meta"
    assert moved.attention_mask.device.type == "meta"
    assert moved.labels.device.type == "meta"
    assert moved.sample_weights.device.type == "meta"
    assert moved.sample_ids == batch.sample_ids
    assert moved.retained_count == batch.retained_count
