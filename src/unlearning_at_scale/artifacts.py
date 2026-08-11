from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResolvedHubArtifact:
    kind: str
    repo_id: str
    requested_revision: str
    resolved_sha: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def directory_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_huggingface(kind: str, repo_id: str, revision: str) -> ResolvedHubArtifact:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install the llm extra to resolve Hugging Face revisions") from exc
    api = HfApi()
    if kind == "model":
        info = api.model_info(repo_id, revision=revision)
    elif kind == "dataset":
        info = api.dataset_info(repo_id, revision=revision)
    else:
        raise ValueError(f"unsupported Hugging Face artifact kind: {kind}")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}@{revision}")
    return ResolvedHubArtifact(kind, repo_id, revision, str(info.sha))


def load_sources(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("artifact source file must contain a mapping")
    return payload


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def freeze_artifacts(source_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source_path = Path(source_path)
    config = load_sources(source_path)
    output: dict[str, Any] = {
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(source_path),
        "huggingface": {},
        "prepared_datasets": {},
        "upstreams": {},
    }
    for name, spec in sorted(config.get("huggingface", {}).items()):
        resolved = resolve_huggingface(str(spec["kind"]), str(spec["repo_id"]), str(spec["revision"]))
        output["huggingface"][name] = asdict(resolved)
    for name, spec in sorted(config.get("prepared_datasets", {}).items()):
        path = Path(spec["path"])
        files = hash_directory(path)
        output["prepared_datasets"][name] = {
            "path": path.as_posix(),
            "files": files,
            "directory_sha256": directory_digest(files),
        }
    for name, spec in sorted(config.get("upstreams", {}).items()):
        expected = str(spec["commit"])
        checkout = Path(spec["checkout"])
        observed = git_head(checkout)
        if observed != expected:
            raise RuntimeError(f"{name} checkout is {observed}, expected pinned commit {expected}")
        output["upstreams"][name] = {
            "repository": str(spec["repository"]),
            "commit": expected,
            "checkout": checkout.as_posix(),
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def verify_lock(source_path: str | Path, lock_path: str | Path, *, verify_prepared_datasets: bool = True, verify_upstreams: bool = True) -> None:
    source_path = Path(source_path)
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("source_manifest_sha256") != sha256_file(source_path):
        raise RuntimeError("artifact source manifest changed after the lock was created")
    for name, item in lock.get("huggingface", {}).items():
        resolved = str(item.get("resolved_sha", ""))
        if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved.lower()):
            raise RuntimeError(f"{name} is not frozen to a full 40-character commit SHA")
    if verify_prepared_datasets:
        for name, item in lock.get("prepared_datasets", {}).items():
            observed = hash_directory(Path(item["path"]))
            if observed != item["files"] or directory_digest(observed) != item["directory_sha256"]:
                raise RuntimeError(f"prepared dataset {name} no longer matches its lock")
    if verify_upstreams:
        for name, item in lock.get("upstreams", {}).items():
            if git_head(Path(item["checkout"])) != item["commit"]:
                raise RuntimeError(f"upstream {name} moved from its pinned commit")


def revision_from_lock(lock_path: str | Path, key: str) -> str:
    lock = json.loads(Path(lock_path).read_text())
    try:
        revision = str(lock["huggingface"][key]["resolved_sha"])
    except KeyError as exc:
        raise KeyError(f"artifact key {key!r} is absent from {lock_path}") from exc
    if len(revision) != 40:
        raise RuntimeError(f"artifact key {key!r} is not frozen to a full commit SHA")
    return revision
