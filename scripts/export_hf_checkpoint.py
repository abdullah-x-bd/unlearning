from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from unlearning_at_scale.artifacts import revision_from_lock, verify_lock
from unlearning_at_scale.modeling import load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an exact runner state as a Hugging Face checkpoint for external evaluation")
    parser.add_argument("config")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", default="locks/artifacts.lock.json")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    args = parser.parse_args()
    verify_lock(args.sources, args.lock)
    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = config["model"]
    revision = revision_from_lock(args.lock, model_cfg["artifact_key"])
    model, tokenizer = load_causal_lm(model_cfg["name"], revision=revision, attention_implementation="eager", disable_dropout=False)
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "SOURCE_STATE.txt").write_text(f"state={Path(args.state).resolve()}\nbase={model_cfg['name']}\nrevision={revision}\n")


if __name__ == "__main__":
    main()
