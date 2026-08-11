from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

RELEASE_DIR = Path("results/releases/tofu-llama32-1b-forget10-2026-08-11")
EXPECTED = {
    "source_workflow_run": 31490644488,
    "model_revision": "9213176726f574b556790deb65791e0c5aa438b6",
    "plan_sha256": "466595193cba55c4cf408b5d5c7d679d6d93bcd73dc11a8b356ac2c716c282da",
    "forget_ids_sha256": "dc1f259e1c930087b81fc2c82ac38345a6ff1097760137bbb366a4f62230dc3d",
    "original_model_sha256": "54c711e9bde77215d9c5def50429f925a382bdcd28150bb87a89a118dd54bc65",
    "original_optimizer_sha256": "7dae55302988b834099f224a6ababcbd97a3411afa1b54e5e9a1739854f8e773",
    "deletion_model_sha256": "067109bfd2e34f1616a8069d04ecd28b4814513332b03957ab917503122aeec3",
    "deletion_optimizer_sha256": "0152484534eab198ebee5a500e77d81451f2cba34ea6baa987ccc5726e20daa1",
}


def main() -> None:
    frozen = json.loads((RELEASE_DIR / "frozen-hashes.json").read_text())
    for key, value in EXPECTED.items():
        if frozen.get(key) != value:
            raise RuntimeError(
                f"frozen release mismatch for {key}: {frozen.get(key)} != {value}"
            )

    token = (
        os.environ.get("HF_TOKEN_PRIMARY")
        or os.environ.get("HF_TOKEN_ALT")
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError("HF token is missing; refusing paid evaluation")

    lock = json.loads(Path("locks/artifacts.lock.json").read_text())
    llama = lock["huggingface"]["llama32_1b_instruct"]
    retain90 = lock["huggingface"]["tofu_llama32_1b_retain90_reference"]
    if llama["resolved_sha"] != frozen["model_revision"]:
        raise RuntimeError("artifact lock and TOFU release disagree on Llama revision")

    for item in (llama, retain90):
        hf_hub_download(
            repo_id=item["repo_id"],
            filename="config.json",
            revision=item["resolved_sha"],
            token=token,
        )

    subprocess.run([sys.executable, "scripts/bootstrap_upstreams.py"], check=True)
    observed = subprocess.check_output(
        ["git", "-C", "external/open-unlearning", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    expected_upstream = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
    if observed != expected_upstream:
        raise RuntimeError(f"OpenUnlearning checkout {observed} != {expected_upstream}")

    for path in (
        Path("scripts/reconstruct_tofu_for_openunlearning.py"),
        Path("scripts/openunlearning_adapter.py"),
    ):
        compile(path.read_text(), str(path), "exec")
    subprocess.run(["bash", "-n", "scripts/run_tofu_openunlearning_eval.sh"], check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/openunlearning_adapter.py",
            "tofu-eval",
            "--checkpoint",
            "/tmp/uas-placeholder-original",
            "--forget-split",
            "forget10",
            "--task-name",
            "uas_preflight_original",
            "--attention-implementation",
            "eager",
            "--output-root",
            "/tmp/uas-openunlearning-preflight",
            "--dry-run",
        ],
        check=True,
    )
    print("TOFU OpenUnlearning zero-cost preflight passed")


if __name__ == "__main__":
    main()
