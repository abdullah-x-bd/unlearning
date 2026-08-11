from pathlib import Path

import yaml


def test_tofu_llama1b_release_is_focused_four_pass_forget10():
    config = yaml.safe_load(
        Path("configs/benchmarks/tofu-llama32-1b-forget10.yaml").read_text()
    )

    assert config["release_mode"] is True
    assert config["materialize_redacted_store"] is True
    assert config["model"]["artifact_key"] == "llama32_1b_instruct"
    assert config["model"]["dtype"] == "bf16"
    assert config["plan"]["microbatch_size"] == 8
    assert config["plan"]["grad_accum_steps"] == 4
    assert config["plan"]["epochs"] == 5
    assert config["checkpoint_every"] == 0
    assert config["replay_policies"] == ["slot_mask"]
    assert config["run_repacked_baseline"] is False

    scenarios = config["forget_scenarios"]
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "tofu-forget10"
    assert scenarios[0]["strategy"] == "id_file"
    assert scenarios[0]["path"].endswith("forget10_ids.txt")


def test_tofu_llama1b_full_workflow_has_budget_access_and_cleanup_guards():
    workflow = Path(".github/workflows/runpod-tofu-llama1b-full.yml").read_text()

    assert 'MAX_HOURLY_COST: "0.70"' in workflow
    assert 'MAX_RUNTIME_MINUTES: "240"' in workflow
    assert '--gpu-types "NVIDIA RTX A6000"' in workflow
    assert "HF_TOKEN_PRIMARY" in workflow
    assert "hf_hub_download" in workflow
    assert "runpod_control.py delete" in workflow
    assert "sleep 16200" in workflow
    assert "python scripts/run_release.py configs/benchmarks/tofu-llama32-1b-forget10.yaml" in workflow
    assert "tofu-llama1b-full-evidence.tar.gz" in workflow
