import json

import numpy as np

from unlearning_at_scale.dataset import TokenStore, materialize_redacted_store


def test_materialized_store_has_no_forgotten_row_and_slot_replay_still_works(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "ids.json").write_text(json.dumps(["a", "b", "c"]))
    np.save(source / "input_ids.npy", np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int64))
    np.save(source / "attention_mask.npy", np.ones((3, 2), dtype=np.int64))
    store = TokenStore(source, dummy_token_id=0)
    redacted_dir = tmp_path / "redacted"
    manifest = materialize_redacted_store(store, {"b"}, redacted_dir)
    redacted = TokenStore(redacted_dir, dummy_token_id=0)
    assert "b" not in redacted.id_to_row
    assert manifest["forgotten_ids_present"] is False
    batch = redacted.get_batch(["a", "b", "c"], {"b"}, "slot_mask")
    assert batch.input_ids[1].tolist() == [0, 0]
    assert batch.sample_weights.tolist() == [1.0, 0.0, 1.0]
