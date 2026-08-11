from pathlib import Path


def test_openunlearning_workflow_is_guarded_and_does_not_run_baselines():
    workflow = Path(".github/workflows/runpod-tofu-openunlearning-eval.yml").read_text()
    runner = Path("scripts/run_tofu_openunlearning_eval.sh").read_text()
    preflight = Path("scripts/preflight_tofu_openunlearning.py").read_text()
    cpu_preflight = Path(".github/workflows/cpu-preflight.yml").read_text()

    assert 'MAX_HOURLY_COST: "0.70"' in workflow
    assert 'POD_SELF_DESTRUCT_SECONDS: "14400"' in workflow
    assert '--gpu-types "NVIDIA RTX A6000"' in workflow
    assert "runpod_control.py delete" in workflow
    assert "tofu-openunlearning-evidence.tar.gz" in workflow
    assert "scripts/run_tofu_openunlearning_eval.sh" in workflow
    assert "scripts/preflight_tofu_openunlearning.py" in workflow
    assert "runpodctl pod delete \\$RUNPOD_POD_ID" in workflow
    assert "Preserve hash-verified checkpoints after evaluator failure" in workflow
    assert "tofu-openunlearning-recovery.tar" in workflow
    assert "retention-days: 1" in workflow

    assert "tofu_llama32_1b_retain90_reference" in preflight
    assert "open-unlearning/tofu_Llama-3.2-1B-Instruct" not in preflight
    assert "external/open-unlearning[lm-eval]" in preflight
    assert "lm-eval[hf]==0.4.11" in preflight
    assert "torch==2.4.1" in preflight
    assert "https://download.pytorch.org/whl/cpu" in preflight
    assert '"transformers": "4.51.3"' in preflight
    assert '"accelerate": "0.34.2"' in preflight
    assert '"huggingface_hub": "0.36.0"' in preflight
    assert "import transformers" in preflight
    assert "import accelerate" in preflight
    assert "import peft" in preflight
    assert "from lm_eval.models.hf_vlms import HFLM" in preflight

    assert 'RECONSTRUCTION_MINUTES="${RECONSTRUCTION_MINUTES:-90}"' in runner
    assert 'FIRST_EVAL_MINUTES="${FIRST_EVAL_MINUTES:-75}"' in runner
    assert 'SECOND_EVAL_MINUTES="${SECOND_EVAL_MINUTES:-45}"' in runner
    assert runner.count("openunlearning_adapter.py tofu-eval") == 2
    assert runner.count("--attention-implementation eager") == 2
    assert "external/open-unlearning[lm-eval]" in runner
    assert "lm-eval[hf]==0.4.11" in runner
    assert "import transformers" in runner
    assert "import accelerate" in runner
    assert "import peft" in runner
    assert "from lm_eval.models.hf_vlms import HFLM" in runner
    assert "tofu-baselines" not in runner
    assert "src/train.py" not in runner
    assert "Canonical state hash gate passed" in runner

    assert "lm-eval[hf]==0.4.11" in cpu_preflight
    assert "from lm_eval.models.hf_vlms import HFLM" in cpu_preflight


def test_openunlearning_adapter_pins_retain_snapshot_and_eager_backend():
    adapter = Path("scripts/openunlearning_adapter.py").read_text()

    assert '"retain90": "tofu_llama32_1b_retain90_reference"' in adapter
    assert "snapshot_download" in adapter
    assert "revision=item[\"resolved_sha\"]" in adapter
    assert 'default="eager"' in adapter
    assert "model.model_args.attn_implementation" in adapter
    assert "model.tokenizer_args.pretrained_model_name_or_path" in adapter
