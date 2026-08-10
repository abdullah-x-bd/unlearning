from __future__ import annotations

import csv
import json
from pathlib import Path


def collect_run_rows(root: str | Path) -> list[dict]:
    root = Path(root)
    rows: list[dict] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        relative = summary_path.relative_to(root)
        row = {
            "summary_path": str(relative),
            "kind": relative.parts[-2] if len(relative.parts) >= 2 else "root",
        }
        if isinstance(payload, dict):
            for key in [
                "model_sha256",
                "optimizer_sha256",
                "forget_count",
                "earliest_affected_step",
                "checkpoint_step",
            ]:
                if key in payload:
                    row[key] = payload[key]
            stats = payload.get("stats")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    row[f"stats.{key}"] = value
            for comparison_key in [
                "model_comparison",
                "comparison_to_oracle",
                "comparison_to_trace_oracle",
            ]:
                comp = payload.get(comparison_key)
                if isinstance(comp, dict):
                    for key, value in comp.items():
                        row[f"{comparison_key}.{key}"] = value
        rows.append(row)
    return rows


def write_rows_csv(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
