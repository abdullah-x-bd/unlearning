from pathlib import Path


def test_reconstruction_is_two_pass_hash_gated_and_physically_redacted():
    text = Path("scripts/reconstruct_tofu_for_openunlearning.py").read_text()

    assert text.count(".run(") == 2
    assert "materialize_redacted_store" in text
    assert 'redaction.get("forgotten_ids_present") is not False' in text
    assert 'len(forget_ids) != 400' in text
    assert "expected_original_model_sha256" in text
    assert "expected_original_optimizer_sha256" in text
    assert "expected_deletion_model_sha256" in text
    assert "expected_deletion_optimizer_sha256" in text
    assert "expected_plan_sha256" in text
    assert "expected_forget_sha256" in text
    assert "run_experiment" not in text
    assert "policy=\"slot_mask\"" in text
    assert "save_pretrained" in text


def test_openunlearning_eval_defaults_to_explicit_eager_attention():
    text = Path("scripts/openunlearning_adapter.py").read_text()

    assert 'default="eager"' in text
    assert "model.model_args.attn_implementation" in text
    assert "model.tokenizer_args.pretrained_model_name_or_path" in text
    assert '"attention_implementation": args.attention_implementation' in text


def test_frozen_tofu_hashes_are_canonical_release_values():
    import json

    frozen = json.loads(
        Path(
            "results/releases/tofu-llama32-1b-forget10-2026-08-11/frozen-hashes.json"
        ).read_text()
    )

    assert frozen["source_workflow_run"] == 31490644488
    assert frozen["model_revision"] == "9213176726f574b556790deb65791e0c5aa438b6"
    assert frozen["plan_sha256"] == "466595193cba55c4cf408b5d5c7d679d6d93bcd73dc11a8b356ac2c716c282da"
    assert frozen["forget_count"] == 400
    assert frozen["original_model_sha256"] == "54c711e9bde77215d9c5def50429f925a382bdcd28150bb87a89a118dd54bc65"
    assert frozen["deletion_model_sha256"] == "067109bfd2e34f1616a8069d04ecd28b4814513332b03957ab917503122aeec3"
