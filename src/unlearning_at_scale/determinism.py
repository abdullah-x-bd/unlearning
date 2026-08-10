from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def configure_determinism(seed: int, strict: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(strict, warn_only=not strict)


def seed_microbatch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_snapshot() -> dict[str, Any]:
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def write_environment_snapshot(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n")
