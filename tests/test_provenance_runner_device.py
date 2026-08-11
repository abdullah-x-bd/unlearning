from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_provenance_ablations.py"
    spec = importlib.util.spec_from_file_location("run_provenance_ablations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_model_optimizer_moves_model_before_optimizer(monkeypatch):
    module = load_runner_module()

    class DummyModel:
        def __init__(self):
            self.device_seen = None

        def to(self, device):
            self.device_seen = device
            return self

    model = DummyModel()
    tokenizer = object()
    device = torch.device("cpu")

    monkeypatch.setattr(module, "load_causal_lm", lambda *args, **kwargs: (model, tokenizer))

    def fake_create_optimizer(observed_model, **kwargs):
        assert observed_model is model
        assert model.device_seen == device
        return object()

    monkeypatch.setattr(module, "create_optimizer", fake_create_optimizer)

    config = {
        "model": {
            "name": "dummy",
            "revision": "frozen",
            "attention_implementation": "eager",
            "disable_dropout": False,
        },
        "optimizer": {
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
    }

    observed_model, observed_tokenizer, _ = module.new_model_optimizer(config, device)
    assert observed_model is model
    assert observed_tokenizer is tokenizer
    assert model.device_seen == device
