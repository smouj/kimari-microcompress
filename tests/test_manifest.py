"""Tests for KMC manifest serialization/deserialization."""

from kmc.manifest import (
    KMC_MANIFEST_VERSION,
    BlockEntry,
    FileEntry,
    KMCManifest,
)


def test_manifest_roundtrip():
    """Manifest can be serialized to JSON and back without data loss."""
    block = BlockEntry(
        index=0,
        offset=100,
        compressed_size=200,
        original_size=300,
        codec="zstd",
        hash="abc123",
    )
    file_entry = FileEntry(
        path="model.safetensors",
        original_size=300,
        hash="def456",
        block_size=262144,
        blocks=[block],
    )
    manifest = KMCManifest(
        version=KMC_MANIFEST_VERSION,
        files=[file_entry],
        total_original_size=300,
        total_compressed_size=200,
    )

    json_str = manifest.to_json()
    restored = KMCManifest.from_json(json_str)

    assert restored.version == manifest.version
    assert len(restored.files) == 1
    assert restored.files[0].path == "model.safetensors"
    assert restored.files[0].original_size == 300
    assert len(restored.files[0].blocks) == 1
    assert restored.files[0].blocks[0].codec == "zstd"
    assert restored.files[0].blocks[0].hash == "abc123"
    assert restored.total_original_size == 300
    assert restored.total_compressed_size == 200


def test_manifest_bytes_roundtrip():
    """Manifest can be serialized to bytes and back."""
    manifest = KMCManifest()
    raw = manifest.to_bytes()
    restored = KMCManifest.from_bytes(raw)
    assert restored.version == KMC_MANIFEST_VERSION
    assert len(restored.files) == 0


def test_empty_manifest():
    """Empty manifest has correct defaults."""
    manifest = KMCManifest()
    assert manifest.version == KMC_MANIFEST_VERSION
    assert manifest.tool == "kimari-microcompress"
    assert manifest.files == []
    assert manifest.total_original_size == 0
    assert manifest.total_compressed_size == 0


def test_manifest_with_multiple_files():
    """Manifest handles multiple files correctly."""
    files = []
    for i in range(3):
        files.append(
            FileEntry(
                path=f"model_{i}.safetensors",
                original_size=1024 * (i + 1),
                hash=f"hash_{i}",
                block_size=262144,
                blocks=[],
            )
        )

    manifest = KMCManifest(
        files=files,
        total_original_size=sum(f.original_size for f in files),
    )

    json_str = manifest.to_json()
    restored = KMCManifest.from_json(json_str)

    assert len(restored.files) == 3
    assert restored.total_original_size == 1024 + 2048 + 3072
