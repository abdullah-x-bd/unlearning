from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from unlearning_at_scale.audit import evaluate_loss
from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.hotpath import curvature_anti_update, diagonal_fisher
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.state import model_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the explicitly approximate curvature hot path")
    parser.add_argument("config")
    parser.add_argument("--state", required=True)
    parser.add_argument("--forget-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fisher-examples", type=int, default=32)
    parser.add_argument("--audit-examples", type=int, default=128)
    parser.add_argument("--step-size", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    model_cfg = config["model"]
    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    model.load_state_dict(torch.load(args.state, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    store = TokenStore(config["data"]["train_dir"], dummy_token_id=tokenizer.pad_token_id)
    forget_ids = [line.strip() for line in Path(args.forget_ids).read_text().splitlines() if line.strip()]
    forget_ids = [sample_id for sample_id in forget_ids if sample_id in store.id_to_row]
    forget_set = set(forget_ids)
    retain_ids = [sample_id for sample_id in store.ids if sample_id not in forget_set]
    rng = random.Random(seed)
    rng.shuffle(retain_ids)

    before_hash = model_sha256(model)
    before = {
        "forget": evaluate_loss(model, store, forget_ids, device, max_examples=args.audit_examples).to_dict(),
        "retain": evaluate_loss(model, store, retain_ids, device, max_examples=args.audit_examples).to_dict(),
    }
    fisher = diagonal_fisher(model, store, retain_ids, device, max_examples=args.fisher_examples)
    applied_norm = curvature_anti_update(
        model,
        store,
        forget_ids,
        fisher,
        device,
        step_size=args.step_size,
        damping=args.damping,
        trust_radius=args.trust_radius,
    )
    after = {
        "forget": evaluate_loss(model, store, forget_ids, device, max_examples=args.audit_examples).to_dict(),
        "retain": evaluate_loss(model, store, retain_ids, device, max_examples=args.audit_examples).to_dict(),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "hotpath-model-state.pt"
    torch.save(model.state_dict(), state_path)
    payload = {
        "exact": False,
        "method": "diagonal_fisher_curvature_anti_update",
        "before_model_sha256": before_hash,
        "after_model_sha256": model_sha256(model),
        "applied_update_l2": applied_norm,
        "parameters": {
            "fisher_examples": args.fisher_examples,
            "step_size": args.step_size,
            "damping": args.damping,
            "trust_radius": args.trust_radius,
        },
        "before": before,
        "after": after,
        "state_path": str(state_path),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
