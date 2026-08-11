from pathlib import Path

import yaml


def test_pythia_2_8b_release_is_core_four_pass_replication():
    config = yaml.safe_load(Path("configs/pythia-2.8b-scaling.yaml").read_text())

    assert config["model"]["artifact_key"] == "pythia_2_8b"
    assert config["model"]["dtype"] == "bf16"
    assert config["plan"]["microbatch_size"] == 1
    assert config["plan"]["grad_accum_steps"] == 16
    assert config["checkpoint_every"] == 0
    assert config["replay_policies"] == ["slot_mask"]
    assert config["run_repacked_baseline"] is False
    assert config["release_phase_smoke"] is True

    scenarios = config["forget_scenarios"]
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "random-5pct"
    assert scenarios[0]["strategy"] == "random"
    assert float(scenarios[0]["fraction"]) == 0.05


def test_full_runpod_workflow_has_budget_and_cleanup_guards():
    workflow = Path(".github/workflows/runpod-pythia-2.8b-full.yml").read_text()

    assert 'MAX_HOURLY_COST: "2.00"' in workflow
    assert 'MAX_RUNTIME_MINUTES: "250"' in workflow
    assert "runpod_control.py delete" in workflow
    assert "sleep 16200" in workflow
    assert "python scripts/run_release.py configs/pythia-2.8b-scaling.yaml" in workflow
