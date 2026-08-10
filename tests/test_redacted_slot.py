import json

import numpy as np

from unlearning_at_scale.dataset import TokenStore


def test_slot_mask_does_not_read_forgotten_row(tmp_path):
    ids = ["retain", "forget"]
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    np.save(tmp_path / "input_ids.npy", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    np.save(tmp_path / "attention_mask.npy", np.ones((2, 3), dtype=np.int64))
    store = TokenStore(tmp_path, dummy_token_id=0)
    del store.id_to_row["forget"]
    batch = store.get_batch(["retain", "forget"], {"forget"}, "slot_mask")
    assert batch.retained_count == 1
    assert batch.input_ids[1].tolist() == [0, 0, 0]
    assert batch.sample_weights.tolist() == [1.0, 0.0]
