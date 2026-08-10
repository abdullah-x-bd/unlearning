from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .plan import MicrobatchSpec

_RECORD_PREFIX = struct.Struct("<QQfIHH")
_RECORD = struct.Struct("<QQfIHHI")
RECORD_SIZE = _RECORD.size
FLAG_ACCUM_END = 1


def digest_sample_ids(sample_ids: Iterable[str], key: bytes | None = None) -> tuple[int, str]:
    payload = b"\x00".join(item.encode("utf-8") for item in sample_ids)
    if key:
        full = hmac.new(key, payload, hashlib.sha256).digest()
    else:
        full = hashlib.sha256(payload).digest()
    return int.from_bytes(full[:8], "little"), full.hex()


@dataclass(frozen=True)
class WalRecord:
    ids_digest64: int
    seed: int
    lr: float
    optimizer_step: int
    microbatch_len: int
    flags: int
    crc32: int

    @property
    def accumulation_end(self) -> bool:
        return bool(self.flags & FLAG_ACCUM_END)

    def pack(self) -> bytes:
        prefix = _RECORD_PREFIX.pack(
            self.ids_digest64,
            self.seed,
            self.lr,
            self.optimizer_step,
            self.microbatch_len,
            self.flags,
        )
        crc = zlib.crc32(prefix) & 0xFFFFFFFF
        return prefix + struct.pack("<I", crc)

    @classmethod
    def from_spec(cls, spec: MicrobatchSpec, key: bytes | None = None) -> tuple["WalRecord", str]:
        digest64, full_digest = digest_sample_ids(spec.sample_ids, key=key)
        flags = FLAG_ACCUM_END if spec.accumulation_end else 0
        record = cls(
            ids_digest64=digest64,
            seed=spec.seed,
            lr=spec.lr,
            optimizer_step=spec.optimizer_step,
            microbatch_len=len(spec.sample_ids),
            flags=flags,
            crc32=0,
        )
        packed = record.pack()
        return cls(*_RECORD.unpack(packed)), full_digest

    @classmethod
    def unpack(cls, payload: bytes) -> "WalRecord":
        if len(payload) != RECORD_SIZE:
            raise ValueError(f"WAL record must be {RECORD_SIZE} bytes")
        fields = _RECORD.unpack(payload)
        expected = zlib.crc32(payload[: _RECORD_PREFIX.size]) & 0xFFFFFFFF
        if fields[-1] != expected:
            raise ValueError("WAL record CRC mismatch")
        return cls(*fields)


class WalWriter:
    def __init__(self, wal_path: str | Path, manifest_path: str | Path, hmac_key: bytes | None = None):
        self.wal_path = Path(wal_path)
        self.manifest_path = Path(manifest_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.hmac_key = hmac_key
        self._wal = self.wal_path.open("wb")
        self._manifest = self.manifest_path.open("w", encoding="utf-8")
        self._wal_sha = hashlib.sha256()
        self._manifest_sha = hashlib.sha256()
        self._count = 0

    @classmethod
    def from_environment(cls, wal_path: str | Path, manifest_path: str | Path) -> "WalWriter":
        raw_key = os.environ.get("UNLEARNING_WAL_HMAC_KEY")
        return cls(wal_path, manifest_path, raw_key.encode() if raw_key else None)

    def append(self, spec: MicrobatchSpec) -> None:
        record, full_digest = WalRecord.from_spec(spec, key=self.hmac_key)
        packed = record.pack()
        self._wal.write(packed)
        self._wal_sha.update(packed)
        manifest_entry = {
            "record_index": self._count,
            "ids_digest64": f"{record.ids_digest64:016x}",
            "full_digest": full_digest,
            "sample_ids": list(spec.sample_ids),
        }
        line = (json.dumps(manifest_entry, sort_keys=True, separators=(",", ":")) + "\n")
        self._manifest.write(line)
        self._manifest_sha.update(line.encode())
        self._count += 1

    def close(self) -> dict[str, str | int | float]:
        if not self._wal.closed:
            self._wal.flush()
            os.fsync(self._wal.fileno())
            self._wal.close()
        if not self._manifest.closed:
            self._manifest.flush()
            os.fsync(self._manifest.fileno())
            self._manifest.close()
        wal_digest = self._wal_sha.hexdigest()
        manifest_digest = self._manifest_sha.hexdigest()
        self.wal_path.with_suffix(self.wal_path.suffix + ".sha256").write_text(wal_digest + "\n")
        self.manifest_path.with_suffix(self.manifest_path.suffix + ".sha256").write_text(manifest_digest + "\n")
        wal_bytes = self.wal_path.stat().st_size
        manifest_bytes = self.manifest_path.stat().st_size
        total = wal_bytes + manifest_bytes
        return {
            "records": self._count,
            "wal_sha256": wal_digest,
            "manifest_sha256": manifest_digest,
            "record_size": RECORD_SIZE,
            "wal_bytes": wal_bytes,
            "manifest_bytes": manifest_bytes,
            "total_provenance_bytes": total,
            "total_bytes_per_microbatch": total / max(1, self._count),
        }

    def __enter__(self) -> "WalWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class WalReader:
    def __init__(self, wal_path: str | Path, manifest_path: str | Path, hmac_key: bytes | None = None):
        self.wal_path = Path(wal_path)
        self.manifest_path = Path(manifest_path)
        self.hmac_key = hmac_key
        self._manifest = self._load_manifest()

    @classmethod
    def from_environment(cls, wal_path: str | Path, manifest_path: str | Path) -> "WalReader":
        raw_key = os.environ.get("UNLEARNING_WAL_HMAC_KEY")
        return cls(wal_path, manifest_path, raw_key.encode() if raw_key else None)

    @staticmethod
    def _verify_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if checksum_path.exists():
            expected = checksum_path.read_text().strip()
            if expected != actual:
                raise ValueError(f"SHA-256 mismatch for {path.name}")
        return actual

    def _load_manifest(self) -> dict[int, dict]:
        self._verify_file(self.manifest_path)
        mapping: dict[int, dict] = {}
        with self.manifest_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                mapping[int(item["record_index"])] = item
        return mapping

    def verify_sha256(self) -> str:
        return self._verify_file(self.wal_path)

    def records(self) -> Iterator[tuple[int, WalRecord, tuple[str, ...]]]:
        self.verify_sha256()
        with self.wal_path.open("rb") as handle:
            index = 0
            while True:
                payload = handle.read(RECORD_SIZE)
                if not payload:
                    break
                if len(payload) != RECORD_SIZE:
                    raise ValueError("truncated WAL record")
                record = WalRecord.unpack(payload)
                item = self._manifest.get(index)
                if item is None:
                    raise ValueError(f"manifest missing record {index}")
                sample_ids = tuple(item["sample_ids"])
                if len(sample_ids) != record.microbatch_len:
                    raise ValueError(f"manifest length mismatch at record {index}")
                if item["ids_digest64"] != f"{record.ids_digest64:016x}":
                    raise ValueError(f"manifest/WAL digest mismatch at record {index}")
                digest64, full_digest = digest_sample_ids(sample_ids, key=self.hmac_key)
                if digest64 != record.ids_digest64 or full_digest != item["full_digest"]:
                    raise ValueError(f"sample-ID digest verification failed at record {index}")
                yield index, record, sample_ids
                index += 1

    def to_plan(self) -> list[MicrobatchSpec]:
        out: list[MicrobatchSpec] = []
        for index, record, sample_ids in self.records():
            out.append(
                MicrobatchSpec(
                    index=index,
                    sample_ids=sample_ids,
                    seed=record.seed,
                    lr=record.lr,
                    optimizer_step=record.optimizer_step,
                    accumulation_end=record.accumulation_end,
                )
            )
        return out
