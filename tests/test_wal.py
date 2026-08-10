from pathlib import Path

from unlearning_at_scale.plan import MicrobatchSpec
from unlearning_at_scale.wal import RECORD_SIZE, WalReader, WalWriter


def test_wal_round_trip(tmp_path: Path):
    specs = [
        MicrobatchSpec(0, ("a", "b"), 11, 0.0010000000474974513, 0, False),
        MicrobatchSpec(1, ("c",), 12, 0.0010000000474974513, 0, True),
    ]
    writer = WalWriter(tmp_path / "trace.wal", tmp_path / "manifest.jsonl")
    for spec in specs:
        writer.append(spec)
    summary = writer.close()
    assert summary["record_size"] == 32 == RECORD_SIZE
    assert (tmp_path / "trace.wal").stat().st_size == 64
    assert WalReader(tmp_path / "trace.wal", tmp_path / "manifest.jsonl").to_plan() == specs
