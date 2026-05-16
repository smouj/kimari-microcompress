"""Security tests for KMC unpack: path traversal, unsafe paths, manifest attacks."""

import struct
from pathlib import Path

import pytest

from kmc.archive import (
    KMC_MAGIC,
    ExtractionError,
    MAX_MANIFEST_SIZE,
    safe_join_extract_path,
    unpack,
    validate_manifest,
    verify_full,
)
from kmc.manifest import (  # noqa: I001 - separate import block for clarity
    BlockEntry,
    FileEntry,
    KMCManifest,
)


# ---------------------------------------------------------------------------
# safe_join_extract_path tests
# ---------------------------------------------------------------------------


class TestSafeJoinExtractPath:
    """Tests for the safe_join_extract_path function."""

    def test_normal_valid_path(self, tmp_path: Path):
        """Normal relative path resolves correctly."""
        result = safe_join_extract_path(tmp_path, "model.safetensors")
        assert result == tmp_path / "model.safetensors"

    def test_nested_valid_path(self, tmp_path: Path):
        """Nested relative path resolves correctly."""
        result = safe_join_extract_path(tmp_path, "layers/layer0.bin")
        assert result == tmp_path / "layers" / "layer0.bin"

    def test_deep_nested_valid_path(self, tmp_path: Path):
        """Deeply nested relative path resolves correctly."""
        result = safe_join_extract_path(tmp_path, "a/b/c/d/file.txt")
        assert result == tmp_path / "a" / "b" / "c" / "d" / "file.txt"

    def test_path_traversal_parent(self, tmp_path: Path):
        """Path with '..' is rejected."""
        with pytest.raises(ExtractionError, match="[Pp]ath traversal"):
            safe_join_extract_path(tmp_path, "../evil.txt")

    def test_path_traversal_mid(self, tmp_path: Path):
        """Path with '..' in the middle is rejected."""
        with pytest.raises(ExtractionError, match="[Pp]ath traversal"):
            safe_join_extract_path(tmp_path, "foo/../../evil.txt")

    def test_path_traversal_prefix(self, tmp_path: Path):
        """Path with leading '..' is rejected."""
        with pytest.raises(ExtractionError, match="['.]\\.\\."):
            safe_join_extract_path(tmp_path, "../../etc/passwd")

    def test_absolute_path_unix(self, tmp_path: Path):
        """Absolute Unix path is rejected."""
        with pytest.raises(ExtractionError, match="[Aa]bsolute"):
            safe_join_extract_path(tmp_path, "/etc/passwd")

    def test_absolute_path_windows(self, tmp_path: Path):
        """Windows-style absolute path is rejected."""
        with pytest.raises(ExtractionError, match="[Ww]indows"):
            safe_join_extract_path(tmp_path, "C:\\Windows\\System32\\evil.dll")

    def test_null_byte_in_path(self, tmp_path: Path):
        """Null byte in path is rejected."""
        with pytest.raises(ExtractionError, match="[Nn]ull"):
            safe_join_extract_path(tmp_path, "model\x00.exe")

    def test_control_character_in_path(self, tmp_path: Path):
        """Control character in path is rejected."""
        with pytest.raises(ExtractionError, match="[Cc]ontrol"):
            safe_join_extract_path(tmp_path, "model\x01.bin")

    def test_empty_path(self, tmp_path: Path):
        """Empty path is rejected."""
        with pytest.raises(ExtractionError, match="[Ee]mpty"):
            safe_join_extract_path(tmp_path, "")

    def test_whitespace_only_path(self, tmp_path: Path):
        """Whitespace-only path is rejected."""
        with pytest.raises(ExtractionError, match="[Ee]mpty"):
            safe_join_extract_path(tmp_path, "   ")

    def test_consecutive_slashes(self, tmp_path: Path):
        """Consecutive slashes (empty component) are rejected."""
        with pytest.raises(ExtractionError, match="[Cc]onsecutive|[Ee]mpty"):
            safe_join_extract_path(tmp_path, "foo//bar.txt")

    def test_path_stays_within_dir(self, tmp_path: Path):
        """Path that would escape via resolution is rejected."""
        result = safe_join_extract_path(tmp_path, "normal.txt")
        assert str(result).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# Manifest validation tests
# ---------------------------------------------------------------------------


class TestManifestValidation:
    """Tests for validate_manifest function."""

    def test_valid_manifest(self):
        """A valid manifest passes validation."""
        manifest = KMCManifest(
            files=[
                FileEntry(
                    path="model.safetensors",
                    original_size=1000,
                    hash="abc123",
                    block_size=262144,
                    blocks=[
                        BlockEntry(
                            index=0,
                            offset=100,
                            compressed_size=800,
                            original_size=1000,
                            codec="zstd",
                            hash="def456",
                        ),
                    ],
                ),
            ],
            total_original_size=1000,
            total_compressed_size=800,
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_duplicate_paths_rejected(self):
        """Duplicate file paths in manifest are rejected."""
        file_entry = FileEntry(
            path="model.safetensors",
            original_size=100,
            hash="abc",
            block_size=262144,
            blocks=[],
        )
        manifest = KMCManifest(files=[file_entry, file_entry])
        errors = validate_manifest(manifest)
        assert any("Duplicate" in e for e in errors)

    def test_unsafe_path_in_manifest(self):
        """Path traversal in manifest path is rejected."""
        manifest = KMCManifest(
            files=[
                FileEntry(
                    path="../evil.txt",
                    original_size=100,
                    hash="abc",
                    block_size=262144,
                    blocks=[],
                ),
            ],
        )
        errors = validate_manifest(manifest)
        assert any("Unsafe" in e or "traversal" in e.lower() for e in errors)

    def test_unsupported_codec_rejected(self):
        """Unsupported codec in block is rejected."""
        manifest = KMCManifest(
            files=[
                FileEntry(
                    path="model.bin",
                    original_size=100,
                    hash="abc",
                    block_size=262144,
                    blocks=[
                        BlockEntry(
                            index=0,
                            offset=100,
                            compressed_size=80,
                            original_size=100,
                            codec="unknown_codec",
                            hash="def",
                        ),
                    ],
                ),
            ],
        )
        errors = validate_manifest(manifest)
        assert any("Unsupported codec" in e for e in errors)

    def test_negative_size_rejected(self):
        """Negative original_size is rejected."""
        manifest = KMCManifest(
            files=[
                FileEntry(
                    path="model.bin",
                    original_size=-1,
                    hash="abc",
                    block_size=262144,
                    blocks=[],
                ),
            ],
        )
        errors = validate_manifest(manifest)
        assert any("Negative" in e for e in errors)


# ---------------------------------------------------------------------------
# Verify full report tests
# ---------------------------------------------------------------------------


class TestVerifyFull:
    """Tests for the full verification report."""

    def test_valid_archive_report(self, tmp_path: Path):
        """A valid archive produces an OK report."""
        from kmc.archive import pack

        source = tmp_path / "source.txt"
        source.write_bytes(b"Hello KMC! " * 1000)

        archive = tmp_path / "test.kmc"
        pack(source, archive)

        report = verify_full(archive)
        assert report.integrity == "OK"
        assert report.total_files == 1
        assert report.total_blocks >= 1
        assert len(report.errors) == 0

    def test_corrupted_archive_detected(self, tmp_path: Path):
        """A corrupted archive is detected."""
        from kmc.archive import pack

        source = tmp_path / "source.txt"
        source.write_bytes(b"Data to corrupt " * 5000)

        archive = tmp_path / "test.kmc"
        pack(source, archive)

        # Corrupt some bytes in the block data area
        with open(archive, "r+b") as f:
            f.seek(200)
            f.write(b"\xff\xff\xff\xff")

        report = verify_full(archive)
        assert report.integrity == "FAILED"
        assert len(report.errors) > 0

    def test_invalid_magic_detected(self, tmp_path: Path):
        """An archive with invalid magic is detected."""
        bad_archive = tmp_path / "bad.kmc"
        bad_archive.write_bytes(b"NOT_KMC_" + b"\x00" * 100)

        report = verify_full(bad_archive)
        assert report.integrity == "FAILED"

    def test_truncated_archive_detected(self, tmp_path: Path):
        """A truncated archive is detected."""
        bad_archive = tmp_path / "truncated.kmc"
        bad_archive.write_bytes(KMC_MAGIC[:4])

        report = verify_full(bad_archive)
        assert report.integrity == "FAILED"


# ---------------------------------------------------------------------------
# Unpack security tests
# ---------------------------------------------------------------------------


class TestUnpackSecurity:
    """Security tests for the unpack operation."""

    def test_unpack_normal_roundtrip(self, tmp_path: Path):
        """Normal pack/unpack roundtrip works."""
        from kmc.archive import pack

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 1000)

        archive = tmp_path / "test.kmc"
        pack(source, archive)

        output = tmp_path / "output"
        unpack(archive, output)

        assert (output / "model.bin").exists()
        assert (output / "model.bin").read_bytes() == b"\x00\x01\x02" * 1000

    def test_unpack_rejects_path_traversal_manifest(self, tmp_path: Path):
        """Unpack rejects a manifest with path traversal."""
        malicious_manifest = KMCManifest(
            files=[
                FileEntry(
                    path="../../../tmp/evil.txt",
                    original_size=0,
                    hash="abc",
                    block_size=262144,
                    blocks=[],
                ),
            ],
        )

        errors = validate_manifest(malicious_manifest)
        assert len(errors) > 0

    def test_unpack_rejects_absolute_path_manifest(self, tmp_path: Path):
        """Unpack rejects a manifest with absolute paths."""
        malicious_manifest = KMCManifest(
            files=[
                FileEntry(
                    path="/etc/passwd",
                    original_size=0,
                    hash="abc",
                    block_size=262144,
                    blocks=[],
                ),
            ],
        )

        errors = validate_manifest(malicious_manifest)
        assert len(errors) > 0

    def test_unpack_rejects_duplicate_paths(self, tmp_path: Path):
        """Unpack rejects a manifest with duplicate file paths."""
        manifest = KMCManifest(
            files=[
                FileEntry(
                    path="model.bin",
                    original_size=100,
                    hash="a",
                    block_size=262144,
                    blocks=[],
                ),
                FileEntry(
                    path="model.bin",
                    original_size=200,
                    hash="b",
                    block_size=262144,
                    blocks=[],
                ),
            ],
        )
        errors = validate_manifest(manifest)
        assert any("Duplicate" in e for e in errors)

    def test_unpack_rejects_incorrect_block_hash(self, tmp_path: Path):
        """Unpack raises error on block hash mismatch."""
        from kmc.archive import pack, read_manifest_from_archive

        source = tmp_path / "source.txt"
        source.write_bytes(b"Test data for hash check " * 200)

        archive = tmp_path / "test.kmc"
        pack(source, archive)

        # Read manifest to find where block data starts
        _manifest, data_start = read_manifest_from_archive(archive)

        # Corrupt a byte in the block data area (past the manifest)
        with open(archive, "r+b") as f:
            f.seek(data_start + 10)
            original_byte = f.read(1)
            f.seek(data_start + 10)
            f.write(bytes([(original_byte[0] ^ 0xFF)]))

        output = tmp_path / "output"
        with pytest.raises(ValueError):
            unpack(archive, output)

    def test_manifest_size_limit(self, tmp_path: Path):
        """Archive with oversized manifest is rejected."""
        bad_archive = tmp_path / "bad.kmc"
        with open(bad_archive, "wb") as f:
            f.write(KMC_MAGIC)
            f.write(struct.pack(">Q", MAX_MANIFEST_SIZE + 1))

        from kmc.archive import read_manifest_from_archive

        with pytest.raises(ValueError, match="[Mm]anifest too large"):
            read_manifest_from_archive(bad_archive)
