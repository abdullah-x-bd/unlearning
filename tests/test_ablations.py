from unlearning_at_scale.ablations import ablate_plan, supported_ablations
from unlearning_at_scale.plan import build_plan


def test_each_provenance_ablation_changes_plan():
    plan = build_plan(
        ["a", "b", "c", "d", "e", "f"],
        microbatch_size=2,
        grad_accum_steps=2,
        epochs=1,
        shuffle_seed=1,
        rng_seed=2,
        peak_lr=0.01,
        schedule="constant",
        shuffle=False,
    )
    for name in supported_ablations():
        altered = ablate_plan(plan, name)
        assert altered != plan
        assert len(altered) == len(plan)
