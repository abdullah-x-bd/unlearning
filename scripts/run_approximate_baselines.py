from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from unlearning_at_scale.audit import evaluate_loss
from unlearning_at_scale.baselines import precompute_reference_nll, run_approximate_baseline
from unlearning_at_scale.config import load_config
from unlearning_at_scale.dataset import TokenStore
from unlearning_at_scale.determinism import configure_determinism
from unlearning_at_scale.modeling import load_causal_lm
from unlearning_at_scale.state import model_sha256
from unlearning_at_scale.training import create_optimizer


def load_model(config: dict, state_path: Path):
    model_cfg = config["model"]
    model, tokenizer = load_causal_lm(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        attention_implementation=model_cfg.get("attention_implementation", "eager"),
        disable_dropout=bool(model_cfg.get("disable_dropout", False)),
    )
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA, GradDiff, and NPO against the same deletion request")
    parser.add_argument("config")
    parser.add_argument("--state", required=True, help="Original trained model state before unlearning")
    parser.add_argument("--forget-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=["ga", "grad_diff", "npo"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--audit-examples", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    configure_determinism(seed, strict=bool(config.get("strict_determinism", True)))
    model_cfg = config["model"]
    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    state_path = Path(args.state)
    forget_ids = [line.strip() for line in Path(args.forget_ids).read_text().splitlines() if line.strip()]

    probe_model, probe_tokenizer = load_model(config, state_path)
    store = TokenStore(config["data"]["train_dir"], dummy_token_id=probe_tokenizer.pad_token_id)
    forget_ids = [sample_id for sample_id in forget_ids if sample_id in store.id_to_row]
    forget_set = set(forget_ids)
    retain_ids = [sample_id for sample_id in store.ids if sample_id not in forget_set]
    rng = random.Random(seed)
    rng.shuffle(retain_ids)
    reference_nll = None
    if any(method.lower().replace("-", "_") == "npo" for method in args.methods):
        probe_model.to(device)
        reference_nll = precompute_reference_nll(probe_model, store, forget_ids, device, args.batch_size)
    del probe_model, probe_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, method in enumerate(args.methods):
        model, tokenizer = load_model(config, state_path)
        model.to(device)
        before = {
            "model_sha256": model_sha256(model),
            "forget_loss": evaluate_loss(model, store, forget_ids, device, max_examples=args.audit_examples).to_dict(),
            "retain_loss": evaluate_loss(model, store, retain_ids, device, max_examples=args.audit_examples).to_dict(),
        }
        opt_cfg = config["optimizer"]
        optimizer = create_optimizer(
            model,
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            foreach=bool(opt_cfg.get("foreach", False)),
            fused=bool(opt_cfg.get("fused", False)),
        )
        stats = run_approximate_baseline(
            method,
            model,
            optimizer,
            store,
            forget_ids,
            retain_ids,
            device,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed + 1000 + index,
            retain_weight=args.retain_weight,
            beta=args.beta,
            reference_nll=reference_nll,
        )
        method_dir = output / method
        method_dir.mkdir(parents=True, exist_ok=True)
        state_out = method_dir / "final-model-state.pt"
        torch.save(model.state_dict(), state_out)
        after = {
            "model_sha256": model_sha256(model),
            "forget_loss": evaluate_loss(model, store, forget_ids, device, max_examples=args.audit_examples).to_dict(),
            "retain_loss": evaluate_loss(model, store, retain_ids, device, max_examples=args.audit_examples).to_dict(),
        }
        payload = {
            "method": method,
            "exact": False,
            "training": stats.to_dict(),
            "before": before,
            "after": after,
            "state_path": str(state_out),
        }
        (method_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        rows.append(payload)
        del model, tokenizer, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
