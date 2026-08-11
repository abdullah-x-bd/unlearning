import torch

from unlearning_at_scale.determinism import resolve_cuda_index


def test_resolve_cuda_index_uses_explicit_index():
    assert resolve_cuda_index(torch.device("cuda:3")) == 3


def test_resolve_cuda_index_uses_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert resolve_cuda_index(torch.device("cuda")) == 2
