from __future__ import annotations

from pathlib import Path


def attach_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("install the LLM dependencies with: pip install -e '.[llm]'") from exc

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def save_cohort_adapter(model, directory: str | Path) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
