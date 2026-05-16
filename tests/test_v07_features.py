"""Tests for v0.7 features: indexes, KMCReader, selective extraction, kmc list, partial access."""

import json
import os
from pathlib import Path

from kmc.archive import pack, read_manifest_from_archive, unpack, verify_full, verify_quick
from kmc.index import BlockIndex, FileIndex, TensorIndex
from kmc.manifest import KMCManifest, KMC_MANIFEST_VERSION, BlockEntry
from kmc.reader import KMCReader


# ===========================================================================
# Helper: create a simple archive for testing
# ===========================================================================


def _create_test_archive(tmp_path: Path, tensor_aware: bool = False) -> Path:
    """Create a simple test archive and return its path."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "gpt2"}')
    (source / "tokenizer.json").write_text('{"version": 1}')
    (source / "data.bin").write_bytes(os.urandom(500))

    output = tmp_path / "test.kmc"
    pack(source, output, tensor_aware=tensor_aware)
    return output


# ===========================================================================
# Block index tests
# ===========================================================================


class TestBlockIndex:
    """Tests for BlockIndex."""

    def test_block_index_from_manifest(self, tmp_path: Path) -> None:
        """BlockIndex can be built from a manifest."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        assert index.total_blocks > 0

    def test_block_index_has_correct_file_paths(self, tmp_path: Path) -> None:
        """Block index entries have correct file paths."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        file_paths = set(b.file_path for b in index.all_blocks)
        assert "config.json" in file_paths
        assert "tokenizer.json" in file_paths
        assert "data.bin" in file_paths

    def test_block_index_get_by_id(self, tmp_path: Path) -> None:
        """Block index can look up blocks by ID."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        block = index.get_by_id(0)
        assert block is not None
        assert block.block_id == 0

    def test_block_index_get_blocks_for_file(self, tmp_path: Path) -> None:
        """Block index can retrieve blocks for a specific file."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        blocks = index.get_blocks_for_file("config.json")
        assert len(blocks) > 0
        assert all(b.file_path == "config.json" for b in blocks)

    def test_block_index_nonexistent_file(self, tmp_path: Path) -> None:
        """Block index returns empty list for nonexistent file."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        assert index.get_blocks_for_file("nonexistent.txt") == []

    def test_block_index_nonexistent_id(self, tmp_path: Path) -> None:
        """Block index returns None for nonexistent block ID."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = BlockIndex.from_manifest(manifest, archive)
        assert index.get_by_id(9999) is None

    def test_block_offsets_are_set(self, tmp_path: Path) -> None:
        """V0.7 archives have archive_offset set on blocks."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        for f in manifest.files:
            for b in f.blocks:
                assert b.archive_offset > 0, f"Block {b.index} of {f.path} has no archive_offset"

    def test_block_index_reconstructs_old_offsets(self, tmp_path: Path) -> None:
        """BlockIndex reconstructs offsets for manifests without archive_offset."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        # Clear archive_offsets to simulate old manifest
        for f in manifest.files:
            for b in f.blocks:
                b.archive_offset = 0

        index = BlockIndex.from_manifest(manifest, archive)
        # Should still have blocks with computed offsets
        assert index.total_blocks > 0
        for block in index.all_blocks:
            assert block.archive_offset > 0


# ===========================================================================
# File index tests
# ===========================================================================


class TestFileIndex:
    """Tests for FileIndex."""

    def test_file_index_from_manifest(self, tmp_path: Path) -> None:
        """FileIndex can be built from a manifest."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        assert index.total_files == 3

    def test_file_index_list_files(self, tmp_path: Path) -> None:
        """FileIndex lists all files."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        files = index.list_files()
        assert "config.json" in files
        assert "tokenizer.json" in files
        assert "data.bin" in files

    def test_file_index_get(self, tmp_path: Path) -> None:
        """FileIndex can look up a file by path."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        loc = index.get("config.json")
        assert loc is not None
        assert loc.path == "config.json"
        assert loc.size > 0
        assert len(loc.sha256) == 64

    def test_file_index_nonexistent(self, tmp_path: Path) -> None:
        """FileIndex returns None for nonexistent file."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        assert index.get("nonexistent.txt") is None

    def test_file_index_match_pattern(self, tmp_path: Path) -> None:
        """FileIndex can match files against a glob pattern."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        json_files = index.match_pattern("*.json")
        paths = [f.path for f in json_files]
        assert "config.json" in paths
        assert "tokenizer.json" in paths
        assert "data.bin" not in paths

    def test_file_index_match_tokenizer_pattern(self, tmp_path: Path) -> None:
        """FileIndex matches 'tokenizer*' pattern."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        matched = index.match_pattern("tokenizer*")
        paths = [f.path for f in matched]
        assert "tokenizer.json" in paths

    def test_file_index_block_ids(self, tmp_path: Path) -> None:
        """File index entries have block IDs."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        index = FileIndex.from_manifest(manifest)
        loc = index.get("data.bin")
        assert loc is not None
        assert len(loc.block_ids) > 0


# ===========================================================================
# Tensor index tests
# ===========================================================================


class TestTensorIndex:
    """Tests for TensorIndex."""

    def test_tensor_index_empty_for_non_tensor_aware(self, tmp_path: Path) -> None:
        """TensorIndex is empty for non-tensor-aware archives."""
        archive = _create_test_archive(tmp_path, tensor_aware=False)
        manifest, _ = read_manifest_from_archive(archive)

        index = TensorIndex.from_manifest(manifest)
        assert index.total_tensors == 0
        assert not index.available

    def test_tensor_index_available_property(self, tmp_path: Path) -> None:
        """TensorIndex.available reflects whether tensors are indexed."""
        archive = _create_test_archive(tmp_path, tensor_aware=False)
        manifest, _ = read_manifest_from_archive(archive)

        index = TensorIndex.from_manifest(manifest)
        assert not index.available

    def test_tensor_index_list_tensors_empty(self, tmp_path: Path) -> None:
        """TensorIndex lists no tensors for non-tensor-aware archive."""
        archive = _create_test_archive(tmp_path, tensor_aware=False)
        manifest, _ = read_manifest_from_archive(archive)

        index = TensorIndex.from_manifest(manifest)
        assert index.list_tensors() == []


# ===========================================================================
# KMCReader tests
# ===========================================================================


class TestKMCReader:
    """Tests for KMCReader partial-access API."""

    def test_reader_open_and_close(self, tmp_path: Path) -> None:
        """KMCReader can open and close an archive."""
        archive = _create_test_archive(tmp_path)
        reader = KMCReader(archive)
        assert reader.manifest is not None
        reader.close()

    def test_reader_context_manager(self, tmp_path: Path) -> None:
        """KMCReader works as a context manager."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            assert reader.manifest is not None

    def test_reader_file_not_found(self, tmp_path: Path) -> None:
        """KMCReader raises FileNotFoundError for missing archive."""
        import pytest

        with pytest.raises(FileNotFoundError):
            KMCReader(tmp_path / "nonexistent.kmc")

    def test_reader_invalid_archive(self, tmp_path: Path) -> None:
        """KMCReader raises ValueError for invalid archive."""
        import pytest

        bad_file = tmp_path / "bad.kmc"
        bad_file.write_bytes(b"not a kmc file")

        with pytest.raises(ValueError):
            KMCReader(bad_file)

    def test_reader_list_files(self, tmp_path: Path) -> None:
        """KMCReader can list files in the archive."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            files = reader.list_files()
            assert "config.json" in files
            assert "tokenizer.json" in files
            assert "data.bin" in files

    def test_reader_list_tensors(self, tmp_path: Path) -> None:
        """KMCReader can list tensors (empty for non-tensor-aware)."""
        archive = _create_test_archive(tmp_path, tensor_aware=False)
        with KMCReader(archive) as reader:
            tensors = reader.list_tensors()
            assert isinstance(tensors, list)

    def test_reader_get_manifest(self, tmp_path: Path) -> None:
        """KMCReader can return the manifest."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            manifest = reader.get_manifest()
            assert isinstance(manifest, KMCManifest)
            assert manifest.version == 6

    def test_reader_get_file_info(self, tmp_path: Path) -> None:
        """KMCReader can return file metadata."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            info = reader.get_file_info("config.json")
            assert info is not None
            assert info.path == "config.json"
            assert info.size > 0
            assert len(info.sha256) == 64

    def test_reader_get_file_info_nonexistent(self, tmp_path: Path) -> None:
        """KMCReader returns None for nonexistent file."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            assert reader.get_file_info("nonexistent.txt") is None

    def test_reader_read_file(self, tmp_path: Path) -> None:
        """KMCReader can read a file from the archive."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            data = reader.read_file("config.json")
            assert data == b'{"model_type": "gpt2"}'

    def test_reader_read_file_nonexistent(self, tmp_path: Path) -> None:
        """KMCReader raises FileNotFoundError for missing file."""
        archive = _create_test_archive(tmp_path)
        import pytest

        with KMCReader(archive) as reader:
            with pytest.raises(FileNotFoundError):
                reader.read_file("nonexistent.txt")

    def test_reader_read_file_range(self, tmp_path: Path) -> None:
        """KMCReader can read a byte range from a file."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            data = reader.read_file_range("config.json", offset=0, length=10)
            assert len(data) == 10
            # config.json content is '{"model_type": "gpt2"}'
            assert data == b'{"model_ty'

    def test_reader_read_file_range_offset(self, tmp_path: Path) -> None:
        """KMCReader read_file_range with non-zero offset."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            data = reader.read_file_range("config.json", offset=2, length=5)
            assert data == b"model"

    def test_reader_read_file_range_negative_offset(self, tmp_path: Path) -> None:
        """KMCReader read_file_range rejects negative offset."""
        archive = _create_test_archive(tmp_path)
        import pytest

        with KMCReader(archive) as reader:
            with pytest.raises(ValueError):
                reader.read_file_range("config.json", offset=-1, length=5)

    def test_reader_read_file_range_beyond_end(self, tmp_path: Path) -> None:
        """KMCReader read_file_range returns empty for offset beyond file."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            info = reader.get_file_info("config.json")
            data = reader.read_file_range("config.json", offset=info.size + 100, length=10)
            assert data == b""

    def test_reader_extract_file(self, tmp_path: Path) -> None:
        """KMCReader can extract a file to disk."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "extracted"

        with KMCReader(archive) as reader:
            out_path = reader.extract_file("config.json", out_dir)
            assert out_path.exists()
            assert out_path.read_text() == '{"model_type": "gpt2"}'

    def test_reader_read_binary_file(self, tmp_path: Path) -> None:
        """KMCReader can read binary files correctly."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            data = reader.read_file("data.bin")
            assert len(data) == 500

    def test_reader_get_tensor_info_nonexistent(self, tmp_path: Path) -> None:
        """KMCReader returns None for nonexistent tensor."""
        archive = _create_test_archive(tmp_path)
        with KMCReader(archive) as reader:
            assert reader.get_tensor_info("nonexistent.weight") is None


# ===========================================================================
# Manifest v6 tests
# ===========================================================================


class TestManifestV6:
    """Tests for manifest v6 (v0.7) features."""

    def test_manifest_version_is_6(self) -> None:
        """KMC_MANIFEST_VERSION is 6."""
        assert KMC_MANIFEST_VERSION == 6

    def test_default_index_field(self) -> None:
        """Default index field is empty dict."""
        m = KMCManifest()
        assert m.index == {}

    def test_index_in_json(self) -> None:
        """Index field appears in JSON serialization."""
        m = KMCManifest(index={"version": 1, "has_block_offsets": True})
        j = m.to_json()
        data = json.loads(j)
        assert data["index"]["version"] == 1
        assert data["index"]["has_block_offsets"] is True

    def test_index_roundtrip(self) -> None:
        """Index field survives JSON roundtrip."""
        m = KMCManifest(index={"version": 1, "has_block_offsets": True, "has_file_index": True})
        j = m.to_json()
        m2 = KMCManifest.from_json(j)
        assert m2.index["version"] == 1
        assert m2.index["has_block_offsets"] is True

    def test_archive_offset_in_block(self) -> None:
        """BlockEntry has archive_offset field."""
        b = BlockEntry(
            index=0,
            offset=100,
            compressed_size=50,
            original_size=200,
            codec="raw",
            hash="abc",
            archive_offset=100,
        )
        assert b.archive_offset == 100

    def test_archive_offset_default_zero(self) -> None:
        """BlockEntry archive_offset defaults to 0."""
        b = BlockEntry(
            index=0,
            offset=100,
            compressed_size=50,
            original_size=200,
            codec="raw",
            hash="abc",
        )
        assert b.archive_offset == 0

    def test_pack_sets_index_metadata(self, tmp_path: Path) -> None:
        """Packing sets index metadata in manifest."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        assert manifest.index.get("version") == 1
        assert manifest.index.get("has_block_offsets") is True
        assert manifest.index.get("has_file_index") is True

    def test_pack_sets_archive_offsets(self, tmp_path: Path) -> None:
        """Packing sets archive_offset on all blocks."""
        archive = _create_test_archive(tmp_path)
        manifest, _ = read_manifest_from_archive(archive)

        for f in manifest.files:
            for b in f.blocks:
                assert b.archive_offset > 0

    def test_backward_compat_v5_manifest_reads(self) -> None:
        """V5 manifest (without index) reads with empty index."""
        v5_json = json.dumps(
            {
                "version": 5,
                "tool": "kimari-microcompress",
                "tool_version": "0.6.0-alpha",
                "created_at": "2025-01-01",
                "total_original_size": 5000,
                "total_compressed_size": 2500,
                "parallelism": {"created_with_jobs": 2},
                "files": [],
            }
        )
        m = KMCManifest.from_json(v5_json)
        assert m.version == 5
        assert m.index == {}
        assert m.parallelism["created_with_jobs"] == 2

    def test_backward_compat_v4_manifest_reads(self) -> None:
        """V4 manifest (without index or parallelism) reads correctly."""
        v4_json = json.dumps(
            {
                "version": 4,
                "tool": "kimari-microcompress",
                "tool_version": "0.5.0-alpha",
                "created_at": "2025-01-01",
                "total_original_size": 5000,
                "total_compressed_size": 2500,
                "artifact_type": "lora_adapter",
                "files": [],
            }
        )
        m = KMCManifest.from_json(v4_json)
        assert m.version == 4
        assert m.index == {}
        assert m.parallelism == {}

    def test_verify_passes_v6_archive(self, tmp_path: Path) -> None:
        """Verify passes for v6 archive."""
        archive = _create_test_archive(tmp_path)

        report = verify_full(archive)
        assert report.integrity == "OK"

    def test_verify_quick_passes_v6_archive(self, tmp_path: Path) -> None:
        """Quick verify passes for v6 archive."""
        archive = _create_test_archive(tmp_path)

        report = verify_quick(archive)
        assert report.integrity == "OK"


# ===========================================================================
# Selective extraction CLI tests
# ===========================================================================


class TestSelectiveExtraction:
    """Tests for selective extraction features."""

    def test_unpack_list(self, tmp_path: Path) -> None:
        """kmc unpack --list lists available files."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(tmp_path / "out"), "--list",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config.json" in result.stdout

    def test_unpack_list_json(self, tmp_path: Path) -> None:
        """kmc unpack --list --json outputs JSON."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(tmp_path / "out"),
                "--list", "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "files" in data
        assert "config.json" in data["files"]

    def test_unpack_only_config(self, tmp_path: Path) -> None:
        """kmc unpack --only config.json extracts only that file."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir), "--only", "config.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "config.json").exists()
        assert not (out_dir / "data.bin").exists()

    def test_unpack_only_json_pattern(self, tmp_path: Path) -> None:
        """kmc unpack --only '*.json' extracts matching files."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir), "--only", "*.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "config.json").exists()
        assert (out_dir / "tokenizer.json").exists()
        assert not (out_dir / "data.bin").exists()

    def test_unpack_only_json_output(self, tmp_path: Path) -> None:
        """kmc unpack --only --json outputs structured results."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir),
                "--only", "config.json", "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "extracted" in data
        assert "config.json" in data["extracted"]
        assert data["skipped"] > 0

    def test_unpack_no_matches_error(self, tmp_path: Path) -> None:
        """kmc unpack --only with no matches exits with error."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir), "--only", "nonexistent*",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ===========================================================================
# kmc list command tests
# ===========================================================================


class TestKmcList:
    """Tests for kmc list command."""

    def test_list_command(self, tmp_path: Path) -> None:
        """kmc list shows archive contents."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", str(archive)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config.json" in result.stdout
        assert "tokenizer.json" in result.stdout
        assert "data.bin" in result.stdout

    def test_list_json(self, tmp_path: Path) -> None:
        """kmc list --json outputs structured JSON."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", str(archive), "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "files" in data
        assert any(f["path"] == "config.json" for f in data["files"])

    def test_list_files_flag(self, tmp_path: Path) -> None:
        """kmc list --files shows only files."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", str(archive), "--files"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config.json" in result.stdout

    def test_list_tensors_flag(self, tmp_path: Path) -> None:
        """kmc list --tensors shows tensors."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", str(archive), "--tensors"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_list_nonexistent_archive(self, tmp_path: Path) -> None:
        """kmc list fails for nonexistent archive."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", str(tmp_path / "nonexistent.kmc")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ===========================================================================
# Security tests for partial access
# ===========================================================================


class TestPartialAccessSecurity:
    """Security tests for partial access features."""

    def test_path_traversal_only_flag(self, tmp_path: Path) -> None:
        """--only '../evil' is rejected."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir), "--only", "../evil",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_absolute_path_only_flag(self, tmp_path: Path) -> None:
        """--only '/etc/passwd' is rejected."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "selective"
        import subprocess

        result = subprocess.run(
            [
                "python", "-m", "kmc", "unpack",
                str(archive), str(out_dir), "--only", "/etc/passwd",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_reader_checksum_verification(self, tmp_path: Path) -> None:
        """KMCReader verifies block checksums during read."""
        archive = _create_test_archive(tmp_path)

        # Corrupt a block in the archive
        data = bytearray(archive.read_bytes())
        if len(data) > 200:
            data[-50] ^= 0xFF
        corrupt_archive = tmp_path / "corrupt.kmc"
        corrupt_archive.write_bytes(bytes(data))

        import pytest

        with pytest.raises(ValueError, match="checksum mismatch|hash mismatch|Invalid"):
            with KMCReader(corrupt_archive) as reader:
                reader.read_file("data.bin")

    def test_reader_truncated_archive(self, tmp_path: Path) -> None:
        """KMCReader handles truncated archive gracefully."""
        archive = _create_test_archive(tmp_path)
        data = archive.read_bytes()
        truncated = tmp_path / "truncated.kmc"
        truncated.write_bytes(data[: len(data) // 2])

        import pytest

        with pytest.raises((ValueError, OSError)):
            with KMCReader(truncated) as reader:
                reader.read_file("data.bin")

    def test_reader_tensor_nonexistent(self, tmp_path: Path) -> None:
        """KMCReader raises FileNotFoundError for nonexistent tensor."""
        archive = _create_test_archive(tmp_path)
        import pytest

        with KMCReader(archive) as reader:
            with pytest.raises(FileNotFoundError):
                reader.read_tensor("nonexistent.weight")

    def test_reader_extract_safe_path(self, tmp_path: Path) -> None:
        """KMCReader extract_file uses safe path joining."""
        archive = _create_test_archive(tmp_path)
        out_dir = tmp_path / "safe_out"

        with KMCReader(archive) as reader:
            path = reader.extract_file("config.json", out_dir)
            assert path.is_relative_to(out_dir.resolve())

    def test_reader_unicode_filenames(self, tmp_path: Path) -> None:
        """KMCReader works with unicode filenames."""
        source = tmp_path / "unicode_src"
        source.mkdir()
        (source / "datos_ñ.txt").write_bytes(b"spanish data")

        output = tmp_path / "unicode.kmc"
        pack(source, output)

        with KMCReader(output) as reader:
            files = reader.list_files()
            assert "datos_ñ.txt" in files
            data = reader.read_file("datos_ñ.txt")
            assert data == b"spanish data"


# ===========================================================================
# CLI help tests
# ===========================================================================


class TestV07CLI:
    """Tests for v0.7 CLI commands and flags."""

    def test_list_help(self) -> None:
        """kmc list --help works."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "list", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout
        assert "--files" in result.stdout
        assert "--tensors" in result.stdout

    def test_unpack_help_includes_only(self) -> None:
        """kmc unpack --help shows --only flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "unpack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--only" in result.stdout

    def test_unpack_help_includes_tensor(self) -> None:
        """kmc unpack --help shows --tensor flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "unpack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--tensor" in result.stdout

    def test_unpack_help_includes_list(self) -> None:
        """kmc unpack --help shows --list flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "unpack", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--list" in result.stdout

    def test_bench_help_includes_partial_access(self) -> None:
        """kmc bench --help shows --partial-access flag."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "bench", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--partial-access" in result.stdout

    def test_inspect_shows_partial_access(self, tmp_path: Path) -> None:
        """kmc inspect on a .kmc archive shows partial access info."""
        archive = _create_test_archive(tmp_path)
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "inspect", str(archive)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Partial access" in result.stdout
        assert "Block index" in result.stdout

    def test_main_help_shows_list(self) -> None:
        """kmc --help shows the list command."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "kmc", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "list" in result.stdout.lower()


# ===========================================================================
# Roundtrip test with v0.7 features
# ===========================================================================


class TestV07Roundtrip:
    """Roundtrip tests ensuring v0.7 features don't break existing functionality."""

    def test_pack_unpack_roundtrip(self, tmp_path: Path) -> None:
        """Pack and unpack roundtrip correctly with v0.7."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.json").write_text('{"test": true}')
        (source / "data.bin").write_bytes(os.urandom(1000))

        output = tmp_path / "test.kmc"
        pack(source, output)

        restored = tmp_path / "restored"
        unpack(output, restored)

        assert (restored / "config.json").read_text() == '{"test": true}'
        assert (restored / "data.bin").read_bytes() == (source / "data.bin").read_bytes()

    def test_pack_unpack_parallel_roundtrip(self, tmp_path: Path) -> None:
        """Parallel pack/unpack roundtrip works with v0.7."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(os.urandom(2000))

        output = tmp_path / "test.kmc"
        pack(source, output, jobs=2)

        report = verify_full(output)
        assert report.integrity == "OK"

        restored = tmp_path / "restored"
        unpack(output, restored)
        assert (restored / "data.bin").read_bytes() == (source / "data.bin").read_bytes()

    def test_reader_full_roundtrip(self, tmp_path: Path) -> None:
        """KMCReader reads match original files."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.json").write_text('{"model_type": "gpt2"}')
        (source / "tokenizer.json").write_text('{"version": 1}')
        (source / "data.bin").write_bytes(os.urandom(500))

        output = tmp_path / "test.kmc"
        pack(source, output)

        with KMCReader(output) as reader:
            assert reader.read_file("config.json") == b'{"model_type": "gpt2"}'
            assert reader.read_file("tokenizer.json") == b'{"version": 1}'
            assert reader.read_file("data.bin") == (source / "data.bin").read_bytes()


# ===========================================================================
# Experimental safetensors loader tests
# ===========================================================================


class TestSafetensorsLoader:
    """Tests for the experimental safetensors loader."""

    def test_load_tensor_bytes_import(self) -> None:
        """load_tensor_bytes can be imported."""
        from kmc.loaders.safetensors_loader import load_tensor_bytes

        assert callable(load_tensor_bytes)

    def test_load_tensor_import(self) -> None:
        """load_tensor can be imported."""
        from kmc.loaders.safetensors_loader import load_tensor

        assert callable(load_tensor)

    def test_load_tensor_bytes_nonexistent(self, tmp_path: Path) -> None:
        """load_tensor_bytes raises for nonexistent tensor."""
        from kmc.loaders.safetensors_loader import load_tensor_bytes
        import pytest

        archive = _create_test_archive(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_tensor_bytes(archive, "nonexistent.weight")
