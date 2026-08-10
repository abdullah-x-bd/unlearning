from unlearning_at_scale.duplicates import hamming64, simhash64
from unlearning_at_scale.forget import near_duplicate_closure


def test_simhash_duplicate_closure():
    a = simhash64("alpha beta gamma delta epsilon")
    b = simhash64("alpha beta gamma delta epsilon")
    c = simhash64("totally unrelated sentence about mountains")
    assert hamming64(a, b) == 0
    rows = [
        {"sample_id": "a", "simhash64": a},
        {"sample_id": "b", "simhash64": b},
        {"sample_id": "c", "simhash64": c},
    ]
    closed = near_duplicate_closure(rows, {"a"}, max_hamming=0)
    assert closed == {"a", "b"}
