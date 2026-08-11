import json

import numpy as np

from unlearning_at_scale.dataset import IGNORE_INDEX, TokenStore, materialize_redacted_store


def test_token_store_preserves_optional_labels(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "ids.json").write_text(json.dumps(["a", "b"]))
    np.save(source / "input_ids.npy", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    np.save(source / "attention_mask.npy", np.ones((2, 3), dtype=np.int64))
    np.save(source / "labels.npy", np.array([[IGNORE_INDEX, 2, 3], [IGNORE_INDEX, 5, 6]], dtype=np.int64))
    store = TokenStore(source)
    batch = store.get_batch(["a"])
    assert batch.labels.tolist() == [[IGNORE_INDEX, 2, 3]]

    redacted = tmp_path / "redacted"
    materialize_redacted_store(store, {"b"}, redacted)
    redacted_store = TokenStore(redacted)
    assert redacted_store.labels is not None
    assert redacted_store.ids == ["a"]
