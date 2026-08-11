from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "runpod_control.py"
    spec = importlib.util.spec_from_file_location("runpod_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hourly_cost_prefers_adjusted_price():
    module = load_module()
    assert module.hourly_cost({"adjustedCostPerHr": 0.44, "costPerHr": "0.53"}) == 0.44


def test_hourly_cost_accepts_string_price():
    module = load_module()
    assert module.hourly_cost({"costPerHr": "0.69"}) == 0.69


def test_pod_payload_is_single_gpu_non_interruptible_and_ssh_enabled():
    module = load_module()
    payload = module.pod_payload(
        "test",
        "ssh-ed25519 AAAA test",
        "SECURE",
        ["NVIDIA A40"],
        module.DEFAULT_IMAGE,
    )
    assert payload["gpuCount"] == 1
    assert payload["gpuTypeIds"] == ["NVIDIA A40"]
    assert payload["gpuTypePriority"] == "custom"
    assert payload["interruptible"] is False
    assert payload["ports"] == ["22/tcp"]
    assert payload["env"]["PUBLIC_KEY"].startswith("ssh-ed25519")
    assert payload["allowedCudaVersions"] == ["12.4"]
    assert payload["volumeInGb"] >= 50
