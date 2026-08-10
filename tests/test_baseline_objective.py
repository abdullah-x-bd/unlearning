from types import SimpleNamespace

import torch

from unlearning_at_scale.baselines import sequence_nll


class UniformModel(torch.nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        shape = (*input_ids.shape, self.vocab_size)
        return SimpleNamespace(logits=torch.zeros(shape, dtype=torch.float32))


def test_sequence_nll_for_uniform_model():
    model = UniformModel(5)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    nll, counts = sequence_nll(model, ids, mask)
    expected = 3 * torch.log(torch.tensor(5.0))
    assert torch.allclose(nll[0], expected)
    assert counts[0].item() == 3
