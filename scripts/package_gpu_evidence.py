from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

INCLUDE_SUFFIXES = {".json", ".jsonl", ".wal", ".txt", ".yaml", ".yml", ".toml", ".log"}
EXCLUDED_PARTS = {"checkpoints", "redacted-data", "__pycache__"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".bin", ".safetensors"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.suffix in INCLUDE_SUFFIXES or path.name in {"pyproject.toml"}


def copy_tree_files(source: Path, destination: Path, prefix: str) -> list[Path]:
    copied: list[Path] = []
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file() or not eligible(path.relative_to(source)):
            continue
        target = destination / prefix / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Package compact cryptographically indexed GPU-run evidence")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/pythia-160m.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--locks-dir", default="locks")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uas-gpu-evidence-") as temp_dir:
        root = Path(temp_dir) / "evidence"
        root.mkdir(parents=True)
        copied: list[Path] = []
        copied += copy_tree_files(Path(args.run_dir), root, "run")
        copied += copy_tree_files(Path(args.results_dir), root, "results")
        copied += copy_tree_files(Path(args.locks_dir), root, "locks")

        config = Path(args.config)
        if config.exists():
            target = root / "config" / config.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, target)
            copied.append(target)
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            target = root / "repository" / "pyproject.toml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pyproject, target)
            copied.append(target)

        manifest = {
            path.relative_to(root).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(set(copied))
        }
        manifest_path = root / "EVIDENCE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            archive.add(root, arcname="evidence")

    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
