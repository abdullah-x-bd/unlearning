from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from unlearning_at_scale.artifacts import load_sources, resolve_huggingface


def resolve(config: dict, key: str) -> dict:
    spec = config["huggingface"][key]
    item = resolve_huggingface(str(spec["kind"]), str(spec["repo_id"]), str(spec["revision"]))
    return {
        "kind": item.kind,
        "repo_id": item.repo_id,
        "requested_revision": item.requested_revision,
        "resolved_sha": item.resolved_sha,
    }


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all non-GPU release datasets from immutable remote revisions")
    parser.add_argument("--sources", default="locks/artifact-sources.yaml")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--wikitext-max-records", type=int, default=20000)
    args = parser.parse_args()

    config = load_sources(args.sources)
    resolved = {
        key: resolve(config, key)
        for key in ["pythia_160m", "wikitext", "tofu", "tofu_llama32_1b_full_reference"]
    }

    prepared = config["prepared_datasets"]
    wikitext_dir = Path(prepared["wikitext_trace"]["path"])
    tofu_dir = Path(prepared["tofu_llama32_trace"]["path"])
    if args.clean:
        shutil.rmtree(wikitext_dir, ignore_errors=True)
        shutil.rmtree(tofu_dir, ignore_errors=True)

    run([
        sys.executable,
        "scripts/prepare_dataset.py",
        "--dataset", "Salesforce/wikitext",
        "--subset", "wikitext-103-raw-v1",
        "--revision", resolved["wikitext"]["resolved_sha"],
        "--model", "EleutherAI/pythia-160m",
        "--model-revision", resolved["pythia_160m"]["resolved_sha"],
        "--output", str(wikitext_dir),
        "--max-records", str(args.wikitext_max_records),
        "--sequence-length", "256",
        "--canary-groups", "32",
        "--canary-repetitions", "8",
        "--seed", "2026",
    ])

    run([
        sys.executable,
        "scripts/prepare_tofu_openunlearning.py",
        "--output", str(tofu_dir),
        "--tokenizer", "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "--tokenizer-revision", resolved["tofu_llama32_1b_full_reference"]["resolved_sha"],
        "--dataset-revision", resolved["tofu"]["resolved_sha"],
        "--max-length", "512",
    ])

    output = Path("locks/preparation-resolved.json")
    output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"prepared release datasets; resolved inputs written to {output}")


if __name__ == "__main__":
    main()
