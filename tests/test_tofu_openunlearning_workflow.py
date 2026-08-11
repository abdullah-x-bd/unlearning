from pathlib import Path


def test_openunlearning_workflow_is_guarded_and_does_not_run_baselines():
    workflow = Path(".github/workflows/runpod-tofu-openunlearning-eval.yml").read_text()
    runner = Path("scripts/run_tofu_openunlearning_eval.sh").read_text()

    assert 'MAX_HOURLY_COST: "0.70"' in workflow
    assert 'POD_SELF_DESTRUCT_SECONDS: "14400"' in workflow
    assert '--gpu-types "NVIDIA RTX A6000"' in workflow
    assert "runpod_control.py delete" in workflow
    assert "tofu-openunlearning-evidence.tar.gz" in workflow
    assert "scripts/run_tofu_openunlearning_eval.sh" in workflow
    assert "tofu_llama32_1b_retain90_reference" in workflow
    assert "--attention-implementation eager" in workflow

    assert 'RECONSTRUCTION_MINUTES="${RECONSTRUCTION_MINUTES:-90}"' in runner
    assert 'FIRST_EVAL_MINUTES="${FIRST_EVAL_MINUTES:-75}"' in runner
    assert 'SECOND_EVAL_MINUTES="${SECOND_EVAL_MINUTES:-45}"' in runner
    assert runner.count("openunlearning_adapter.py tofu-eval") == 2
    assert "tofu-baselines" not in runner
    assert "src/train.py" not in runner
    assert "Canonical state hash gate passed" in runner


def test_openunlearning_adapter_pins_retain_snapshot_and_eager_backend():
    adapter = Path("scripts/openunlearning_adapter.py").read_text()

    assert '"retain90": "tofu_llama32_1b_retain90_reference"' in adapter
    assert "snapshot_download" in adapter
    assert "revision=item[\"resolved_sha\"]" in adapter
    assert 'default="eager"' in adapter
    assert "model.model_args.attn_implementation" in adapter
    assert "model.tokenizer_args.pretrained_model_name_or_path" in adapter
