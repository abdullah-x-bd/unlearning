from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "package_gpu_evidence.py"
    spec = importlib.util.spec_from_file_location("package_gpu_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eligibility_excludes_large_state_files_and_checkpoints():
    module = load_module()
    assert module.eligible(Path("summary.json"))
    assert module.eligible(Path("original/trace.wal"))
    assert not module.eligible(Path("original/final-model-state.pt"))
    assert not module.eligible(Path("original/checkpoints/step-000250.json"))


def test_manifest_hash_helper(tmp_path):
    module = load_module()
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"ok": True}) + "\n")
    first = module.sha256_file(path)
    second = module.sha256_file(path)
    assert first == second
    assert len(first) == 64
