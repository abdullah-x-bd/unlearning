from __future__ import annotations

import torch


def load_causal_lm(
    model_name: str,
    revision: str | None = None,
    attention_implementation: str = "eager",
    disable_dropout: bool = False,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the LLM dependencies with: pip install -e '.[llm]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"revision": revision}
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False

    if disable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    return model, tokenizer
