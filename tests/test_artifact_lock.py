import json

from unlearning_at_scale.artifacts import directory_digest, hash_directory, verify_lock


def test_directory_hash_changes_with_content(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("one")
    first = hash_directory(data)
    first_digest = directory_digest(first)
    (data / "a.txt").write_text("two")
    second = hash_directory(data)
    assert first != second
    assert first_digest != directory_digest(second)


def test_verify_lock_rejects_non_full_hub_sha(tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text("schema_version: 1\n")
    import hashlib
    source_hash = hashlib.sha256(sources.read_bytes()).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "source_manifest_sha256": source_hash,
        "huggingface": {
            "bad": {
                "resolved_sha": "abc123",
            }
        },
        "prepared_datasets": {},
        "upstreams": {},
    }))
    try:
        verify_lock(sources, lock, verify_prepared_datasets=False, verify_upstreams=False)
    except RuntimeError as exc:
        assert "40-character" in str(exc)
    else:
        raise AssertionError("short Hub revision was accepted")
