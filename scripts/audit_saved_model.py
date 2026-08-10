from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from unlearning_at_scale.audit import evaluate_canaries, evaluate_loss, loss_membership_auc
from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.modeling import load_causal_lm


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a saved model state on forget, retain, held-out, canary, and membership metrics")
    parser.add_argument("config")
    parser.add_argument("--state", required=True)
    parser.add_argument("--forget-ids", required=True)
    parser.add_argument("--validation-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-retain", type=int, default=256)
    parser.add_argument("--max-membership", type=int, default=128)
    parser.add_argument("--max-canaries", type=int, default=64)
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
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)

    train_store = TokenStore(config["data"]["train_dir"], dummy_token_id=tokenizer.pad_token_id)
    forget_ids = [sample_id for sample_id in read_ids(args.forget_ids) if sample_id in train_store.id_to_row]
    forget_set = set(forget_ids)
    retain_ids = [sample_id for sample_id in train_store.ids if sample_id not in forget_set]
    rng = random.Random(seed)
    rng.shuffle(retain_ids)
    retain_ids = retain_ids[: args.max_retain]

    payload = {
        "forget_loss": evaluate_loss(model, train_store, forget_ids, device, max_examples=args.max_retain).to_dict(),
        "retain_loss": evaluate_loss(model, train_store, retain_ids, device, max_examples=args.max_retain).to_dict(),
    }

    canaries = Path(config["data"]["train_dir"]) / "canaries.json"
    if canaries.exists():
        payload["canaries"] = evaluate_canaries(model, tokenizer, canaries, device, max_groups=args.max_canaries).to_dict()

    validation_dir = args.validation_dir or config.get("data", {}).get("validation_dir")
    if validation_dir:
        validation_store = TokenStore(validation_dir, dummy_token_id=tokenizer.pad_token_id)
        nonmember_ids = list(validation_store.ids)
        rng.shuffle(nonmember_ids)
        payload["heldout_loss"] = evaluate_loss(model, validation_store, nonmember_ids, device, max_examples=args.max_retain).to_dict()
        payload["loss_membership"] = loss_membership_auc(
            model,
            train_store,
            retain_ids,
            validation_store,
            nonmember_ids,
            device,
            max_examples=args.max_membership,
        )

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
