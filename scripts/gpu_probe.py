from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast CUDA/BF16 readiness probe")
    parser.add_argument("--output", default="results/gpu-probe.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the allocated Pod")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one GPU, found {torch.cuda.device_count()}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Allocated GPU does not support BF16")

    device = torch.device("cuda:0")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    left = torch.randn((512, 512), device=device, dtype=torch.float32)
    right = torch.randn((512, 512), device=device, dtype=torch.float32)
    first = left @ right
    second = left @ right
    if not torch.equal(first, second):
        raise RuntimeError("Repeated same-process CUDA matmul was not bit-identical")

    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        smi = f"nvidia-smi query failed: {exc}"

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": props.total_memory,
        "free_memory_bytes_at_probe": free_bytes,
        "reported_total_memory_bytes": total_bytes,
        "bf16_supported": True,
        "same_process_matmul_exact": True,
        "nvidia_smi": smi,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
