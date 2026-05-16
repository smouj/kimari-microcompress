"""Tests for pack/unpack roundtrip integrity."""

import tempfile
from pathlib import Path

from kmc.archive import pack, unpack, verify


def test_roundtrip_single_file():
    """Pack a single file, unpack it, and verify byte-for-byte match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test file with repetitive data (like a model would have)
        source_file = tmpdir / "test_model.bin"
        data = b"ABCD" * 10000  # 40 KB of repetitive data
        source_file.write_bytes(data)

        # Pack
        archive = tmpdir / "test.kmc"
        pack(source_file, archive)

        # Verify
        errors = verify(archive)
        assert errors == [], f"Verification errors: {errors}"

        # Unpack
        restore_dir = tmpdir / "restored"
        unpack(archive, restore_dir)

        # Check roundtrip
        restored_file = restore_dir / "test_model.bin"
        assert restored_file.exists(), "Restored file not found"
        assert restored_file.read_bytes() == data, "Roundtrip data mismatch"


def test_roundtrip_directory():
    """Pack a directory with multiple files, unpack, and verify all match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create source directory
        source_dir = tmpdir / "source"
        source_dir.mkdir()

        (source_dir / "model.safetensors").write_bytes(b"\x00\x01\x02\x03" * 5000)
        (source_dir / "config.json").write_bytes(b'{"model_type": "test"}')

        subdir = source_dir / "layers"
        subdir.mkdir()
        (subdir / "layer0.bin").write_bytes(b"\xff" * 10000)

        # Pack
        archive = tmpdir / "test.kmc"
        pack(source_dir, archive)

        # Verify
        errors = verify(archive)
        assert errors == [], f"Verification errors: {errors}"

        # Unpack
        restore_dir = tmpdir / "restored"
        unpack(archive, restore_dir)

        # Check all files
        for f in source_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(source_dir)
                restored = restore_dir / rel
                assert restored.exists(), f"Missing restored file: {rel}"
                assert restored.read_bytes() == f.read_bytes(), f"Mismatch: {rel}"


def test_archive_is_smaller_with_repetitive_data():
    """Repetitive data should compress well in a .kmc archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        source_file = tmpdir / "repetitive.bin"
        data = b"AAAA" * 50000  # 200 KB of highly repetitive data
        source_file.write_bytes(data)

        archive = tmpdir / "test.kmc"
        pack(source_file, archive)

        # Archive should be significantly smaller
        original_size = source_file.stat().st_size
        archive_size = archive.stat().st_size
        assert archive_size < original_size * 0.5, (
            f"Archive not small enough: {archive_size} vs {original_size}"
        )
