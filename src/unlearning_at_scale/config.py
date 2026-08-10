from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
