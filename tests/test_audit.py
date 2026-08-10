from unlearning_at_scale.audit import rank_auc


def test_rank_auc_orders_member_evidence():
    assert rank_auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert rank_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
