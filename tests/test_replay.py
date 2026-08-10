from unlearning_at_scale.smoke import run_core_smoke


def test_exact_replay_smoke(tmp_path):
    result = run_core_smoke(tmp_path / "run")
    assert result["exact"] is True
    assert result["wal_records"] == 4
