from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def test_create_retries_transient_no_capacity(tmp_path, monkeypatch):
    module = load_module()
    public_key = tmp_path / "id_ed25519.pub"
    public_key.write_text("ssh-ed25519 AAAA test")
    output = tmp_path / "allocation.json"
    post_clouds: list[str] = []

    def fake_api_request(method, path, payload=None):
        if method == "POST" and path == "/pods":
            post_clouds.append(payload["cloudType"])
            if len(post_clouds) <= 2:
                raise RuntimeError("create pod: There are no instances currently available")
            return {"id": "pod-1", "costPerHr": "0.53"}
        if method == "GET" and path == "/pods/pod-1":
            return {
                "publicIp": "127.0.0.1",
                "portMappings": {"22": 2200},
                "gpu": {"displayName": "NVIDIA RTX A6000"},
            }
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(module, "api_request", fake_api_request)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("RUNPOD_AVAILABILITY_RETRIES", "3")
    args = SimpleNamespace(
        public_key_file=str(public_key),
        clouds=["SECURE", "COMMUNITY"],
        name="test-retry",
        gpu_types=["NVIDIA RTX A6000"],
        image=module.DEFAULT_IMAGE,
        max_hourly_cost=0.70,
        wait_seconds=1,
        output=str(output),
    )

    module.create(args)

    assert post_clouds == ["SECURE", "COMMUNITY", "SECURE"]
    payload = __import__("json").loads(output.read_text())
    assert payload["pod_id"] == "pod-1"
    assert payload["cost_per_hr"] == 0.53
