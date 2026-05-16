"""Tests for KMC v0.8.0-alpha features.

Covers:
- Optimized read_file_range (block-level range reads)
- Cross-file deduplication
- Delta compression (experimental)
- GGUF quantized codec
- Manifest v7 fields
- CLI --dedup, --delta-base, inspect flags
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kmc.archive import inspect, pack, unpack, verify_full, verify_quick
from kmc.codecs.base import CodecContext
from kmc.codecs.gguf_quant import GGUFQuantCodec
from kmc.codecs.registry import list_codecs
from kmc.dedup import DedupIndex, DedupPlanner, fingerprint_block_data
from kmc.delta import (
    DeltaCodec,
    DeltaPlanner,
    block_similarity,
)
from kmc.manifest import KMC_MANIFEST_VERSION, BlockEntry, FileEntry, KMCManifest
from kmc.reader import KMCReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_dir(tmp_path: Path, name: str = "model") -> Path:
    """Create a temporary directory with test files."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text('{"model_type": "gpt2"}')
    (d / "tokenizer.json").write_text('{"version": 1}')
    (d / "model.safetensors").write_bytes(b"\x00" * 10000)
    return d


def _create_dir_with_duplicates(tmp_path: Path) -> Path:
    """Create a directory with duplicate files for dedup testing."""
    d = tmp_path / "dup_model"
    d.mkdir(exist_ok=True)
    # Two identical files
    data = b"\x42" * 5000
    (d / "file_a.bin").write_bytes(data)
    (d / "file_b.bin").write_bytes(data)
    (d / "config.json").write_text('{"model": "test"}')
    return d


def _pack_and_verify(source: Path, output: Path, **kwargs: object) -> None:
    """Pack a directory and verify the archive integrity."""
    pack(source, output, **kwargs)
    report = verify_full(output)
    assert report.integrity == "OK", f"Verify failed: {report.errors}"


# ---------------------------------------------------------------------------
# read_file_range optimization
# ---------------------------------------------------------------------------


class TestReadFileRangeOptimized:
    """Tests for block-level range reads in KMCReader."""

    def test_range_within_single_block(self, tmp_path: Path) -> None:
        """Read a small range that fits within one block."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            sf = [f for f in files if f.endswith(".safetensors")][0]
            info = reader.get_file_info(sf)
            assert info is not None

            data = reader.read_file_range(sf, offset=100, length=200)
            assert len(data) == 200
            assert data == b"\x00" * 200

    def test_range_crossing_two_blocks(self, tmp_path: Path) -> None:
        """Read a range that crosses a block boundary."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            sf = [f for f in files if f.endswith(".safetensors")][0]

            # Read a range crossing block boundary at 4096
            data = reader.read_file_range(sf, offset=4000, length=512)
            assert len(data) == 512
            assert data == b"\x00" * 512

    def test_range_crossing_many_blocks(self, tmp_path: Path) -> None:
        """Read a range spanning multiple blocks."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=2048)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            sf = [f for f in files if f.endswith(".safetensors")][0]

            data = reader.read_file_range(sf, offset=1000, length=5000)
            assert len(data) == 5000

    def test_range_offset_zero(self, tmp_path: Path) -> None:
        """Read from the beginning of a file."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            cfg = [f for f in files if f.endswith("config.json")][0]

            data = reader.read_file_range(cfg, offset=0, length=10)
            assert len(data) == 10

    def test_range_offset_past_end(self, tmp_path: Path) -> None:
        """Read from offset beyond file size returns empty bytes."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            cfg = [f for f in files if f.endswith("config.json")][0]
            info = reader.get_file_info(cfg)

            data = reader.read_file_range(cfg, offset=info.size + 100, length=100)
            assert data == b""

    def test_range_negative_offset(self, tmp_path: Path) -> None:
        """Negative offset raises ValueError."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            with pytest.raises(ValueError, match="Negative offset"):
                reader.read_file_range(files[0], offset=-1, length=10)

    def test_range_negative_length(self, tmp_path: Path) -> None:
        """Negative length raises ValueError."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            with pytest.raises(ValueError, match="Negative length"):
                reader.read_file_range(files[0], offset=0, length=-1)

    def test_range_file_not_found(self, tmp_path: Path) -> None:
        """Reading range from nonexistent file raises FileNotFoundError."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            with pytest.raises(FileNotFoundError):
                reader.read_file_range("nonexistent.txt", offset=0, length=10)

    def test_range_length_larger_than_file(self, tmp_path: Path) -> None:
        """Length larger than remaining file returns clamped result."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            cfg = [f for f in files if f.endswith("config.json")][0]
            info = reader.get_file_info(cfg)

            # Request more than file size from offset 0
            data = reader.read_file_range(cfg, offset=0, length=info.size + 1000)
            assert len(data) == info.size

    def test_range_matches_full_read(self, tmp_path: Path) -> None:
        """Range read from 0 to file_size matches read_file."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive, block_size=4096)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            for f_path in files:
                info = reader.get_file_info(f_path)
                full_data = reader.read_file(f_path)
                range_data = reader.read_file_range(f_path, offset=0, length=info.size)
                assert full_data == range_data, f"Mismatch for {f_path}"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for cross-file deduplication."""

    def test_fingerprint_identical_data(self) -> None:
        """Identical data produces identical fingerprints."""
        data = b"hello world"
        fp1 = fingerprint_block_data(data)
        fp2 = fingerprint_block_data(data)
        assert fp1 == fp2

    def test_fingerprint_different_data(self) -> None:
        """Different data produces different fingerprints."""
        fp1 = fingerprint_block_data(b"hello")
        fp2 = fingerprint_block_data(b"world")
        assert fp1 != fp2

    def test_dedup_index_identical_blocks(self) -> None:
        """DedupIndex detects duplicate blocks."""
        idx = DedupIndex()
        data = b"\x42" * 1000

        is_dup_0 = idx.add_block(0, 0, 0, data)
        is_dup_1 = idx.add_block(1, 0, 1, data)

        assert is_dup_0 is False  # First occurrence is not a duplicate
        assert is_dup_1 is True  # Second occurrence IS a duplicate
        assert idx.unique_blocks == 1
        assert idx.deduplicated_blocks == 1
        assert idx.saved_bytes == 1000

    def test_dedup_index_different_blocks(self) -> None:
        """DedupIndex keeps different blocks as unique."""
        idx = DedupIndex()

        idx.add_block(0, 0, 0, b"data_a")
        idx.add_block(1, 0, 1, b"data_b")

        assert idx.unique_blocks == 2
        assert idx.deduplicated_blocks == 0

    def test_dedup_planner(self) -> None:
        """DedupPlanner creates correct plans."""
        planner = DedupPlanner()
        data = b"\x42" * 1000

        planner.add_block(0, data)
        planner.add_block(1, data)  # duplicate
        planner.add_block(2, b"\x99" * 1000)  # unique

        plan = planner.create_plan()
        assert plan.total_blocks == 3
        assert plan.unique_blocks == 2
        assert plan.deduplicated_blocks == 1
        assert plan.is_duplicate(1)
        assert not plan.is_duplicate(0)
        assert not plan.is_duplicate(2)

    def test_pack_with_dedup(self, tmp_path: Path) -> None:
        """Pack with --dedup creates a valid archive."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        manifest = inspect(archive)
        assert manifest.deduplication.get("enabled", False) is True

        # Verify archive integrity
        report = verify_full(archive)
        assert report.integrity == "OK"

    def test_dedup_roundtrip(self, tmp_path: Path) -> None:
        """Dedup archive unpacks to identical files."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        output = tmp_path / "unpacked"
        unpack(archive, output)

        # Files should be identical
        for fname in ["file_a.bin", "file_b.bin", "config.json"]:
            original = (source / fname).read_bytes()
            restored = (output / fname).read_bytes()
            assert original == restored, f"Mismatch for {fname}"

    def test_dedup_verify_full(self, tmp_path: Path) -> None:
        """verify_full works on dedup archives."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        report = verify_full(archive)
        assert report.integrity == "OK"

    def test_dedup_verify_quick(self, tmp_path: Path) -> None:
        """verify_quick works on dedup archives."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        report = verify_quick(archive)
        assert report.integrity == "OK"

    def test_dedup_reader(self, tmp_path: Path) -> None:
        """KMCReader works with dedup archives."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        with KMCReader(archive) as reader:
            files = reader.list_files()
            assert len(files) >= 3

            # Read both duplicate files
            data_a = reader.read_file("file_a.bin")
            data_b = reader.read_file("file_b.bin")
            assert data_a == data_b

    def test_dedup_manifest_stats(self, tmp_path: Path) -> None:
        """Dedup manifest records correct statistics."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        manifest = inspect(archive)
        dedup = manifest.deduplication
        assert dedup.get("enabled") is True
        assert dedup.get("fingerprint") == "sha256"
        assert dedup.get("unique_blocks", 0) > 0
        assert dedup.get("deduplicated_blocks", 0) >= 0


# ---------------------------------------------------------------------------
# Delta compression
# ---------------------------------------------------------------------------


class TestDeltaCompression:
    """Tests for experimental delta compression."""

    def test_block_similarity_identical(self) -> None:
        """Identical blocks have similarity 1.0."""
        data = b"\x42" * 1000
        assert block_similarity(data, data) == 1.0

    def test_block_similarity_different(self) -> None:
        """Different blocks have similarity 0.0."""
        assert block_similarity(b"aaaa", b"bbbb") == 0.0

    def test_block_similarity_different_sizes(self) -> None:
        """Blocks of different sizes have similarity 0.0."""
        assert block_similarity(b"short", b"much longer data") == 0.0

    def test_delta_codec_changed_block(self) -> None:
        """DeltaCodec marks changed blocks correctly."""
        codec = DeltaCodec()
        block = codec.compare_block(0, "hash_a", {1: "hash_b"})
        assert block.is_changed is True

    def test_delta_codec_unchanged_block(self) -> None:
        """DeltaCodec marks matching blocks as unchanged."""
        codec = DeltaCodec()
        block = codec.compare_block(0, "same_hash", {1: "same_hash"})
        assert block.is_changed is False
        assert block.base_block_id == 1

    def test_delta_planner_with_base(self, tmp_path: Path) -> None:
        """DeltaPlanner initializes with a base archive."""
        source = _create_test_dir(tmp_path)
        base_archive = tmp_path / "base.kmc"
        pack(source, base_archive)

        planner = DeltaPlanner(base_archive)
        plan = planner.create_plan()
        assert plan.enabled is True

    def test_delta_planner_no_base(self, tmp_path: Path) -> None:
        """DeltaPlanner with nonexistent base still creates plan."""
        planner = DeltaPlanner(tmp_path / "nonexistent.kmc")
        plan = planner.create_plan()
        assert plan.enabled is True

    def test_pack_with_delta_base(self, tmp_path: Path) -> None:
        """Pack with --delta-base creates valid archive."""
        source = _create_test_dir(tmp_path)
        base_archive = tmp_path / "base.kmc"
        pack(source, base_archive)

        delta_archive = tmp_path / "delta.kmc"
        pack(source, delta_archive, delta_base=base_archive)

        manifest = inspect(delta_archive)
        assert manifest.delta.get("enabled") is True
        assert manifest.delta.get("mode") == "experimental"

    def test_delta_manifest_fields(self, tmp_path: Path) -> None:
        """Delta manifest records correct fields."""
        source = _create_test_dir(tmp_path)
        base_archive = tmp_path / "base.kmc"
        pack(source, base_archive)

        delta_archive = tmp_path / "delta.kmc"
        pack(source, delta_archive, delta_base=base_archive)

        manifest = inspect(delta_archive)
        delta = manifest.delta
        assert "base_archive_sha256" in delta
        assert "base_archive_path_hint" in delta

    def test_delta_verify_full(self, tmp_path: Path) -> None:
        """verify_full works on delta archives."""
        source = _create_test_dir(tmp_path)
        base_archive = tmp_path / "base.kmc"
        pack(source, base_archive)

        delta_archive = tmp_path / "delta.kmc"
        pack(source, delta_archive, delta_base=base_archive)

        report = verify_full(delta_archive)
        assert report.integrity == "OK"

    def test_delta_planner_add_block_with_base_data(self) -> None:
        """DeltaPlanner detects unchanged blocks with explicit base data."""
        planner = DeltaPlanner("/nonexistent/base.kmc")
        data = b"\x42" * 1000

        # Same data as base
        block = planner.add_block_with_base_data(0, data, data, "model.bin", 0)
        assert block.is_changed is False

        # Different data from base
        new_data = b"\x99" * 1000
        block2 = planner.add_block_with_base_data(1, new_data, data, "model.bin", 1)
        assert block2.is_changed is True


# ---------------------------------------------------------------------------
# GGUF codec
# ---------------------------------------------------------------------------


class TestGGUFQuantCodec:
    """Tests for the GGUF quantized block codec."""

    def test_gguf_quant_roundtrip(self) -> None:
        """GGUF codec roundtrip preserves data exactly."""
        codec = GGUFQuantCodec()
        data = b"\x00\x01\x02\x03" * 256  # 1024 bytes
        ctx = CodecContext(original_size=len(data))

        result = codec.compress(data, context=ctx)
        assert result.original_size == len(data)

        # Decompress
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata  # type: ignore[attr-defined]
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data

    def test_gguf_quant_no_floatplane(self) -> None:
        """GGUF codec doesn't apply floatplane to float dtypes."""
        codec = GGUFQuantCodec()
        data = b"\x00\x01\x02\x03" * 256
        ctx = CodecContext(original_size=len(data), dtype="BF16")

        result = codec.compress(data, context=ctx)
        # Should fall back to raw since float dtype detected
        assert result.metadata.get("fallback") == "raw" or result.compressed_size <= len(data)

    def test_gguf_quant_with_quant_dtype(self) -> None:
        """GGUF codec works with quantized dtypes."""
        codec = GGUFQuantCodec()
        data = b"\x00\x01\x02\x03" * 256
        ctx = CodecContext(original_size=len(data), dtype="Q4_0")

        result = codec.compress(data, context=ctx)
        assert result.original_size == len(data)
        # Should have tried compression
        assert "candidates_tried" in result.metadata

    def test_gguf_quant_fallback_on_missing_metadata(self) -> None:
        """GGUF codec handles missing metadata gracefully."""
        codec = GGUFQuantCodec()
        data = b"\x00\x01\x02\x03" * 256
        ctx = CodecContext(original_size=len(data))

        result = codec.compress(data, context=ctx)
        # Even without specific dtype, should produce valid output
        assert result.original_size == len(data)

    def test_gguf_quant_raw_fallback(self) -> None:
        """GGUF codec falls back to raw if compression doesn't help."""
        codec = GGUFQuantCodec()
        # Random-ish data that won't compress well
        data = bytes(range(256)) * 4
        ctx = CodecContext(original_size=len(data))

        result = codec.compress(data, context=ctx)
        # Result should be valid (either compressed or raw fallback)
        assert result.original_size == len(data)

    def test_gguf_quant_in_registry(self) -> None:
        """gguf_quant_block is in the codec registry."""
        codecs = list_codecs()
        assert "gguf_quant_block" in codecs

    def test_gguf_quant_decompress_without_context(self) -> None:
        """GGUF codec raises ValueError without context for decompression."""
        codec = GGUFQuantCodec()
        with pytest.raises(ValueError, match="context"):
            codec.decompress(b"\x00" * 100, context=None)


# ---------------------------------------------------------------------------
# Manifest v7
# ---------------------------------------------------------------------------


class TestManifestV7:
    """Tests for manifest v7 fields."""

    def test_manifest_v7_version(self) -> None:
        """New manifests have version 7."""
        assert KMC_MANIFEST_VERSION == 7

    def test_manifest_v7_new_fields(self) -> None:
        """Manifest v7 has deduplication, delta, runtime_hints fields."""
        m = KMCManifest()
        assert hasattr(m, "deduplication")
        assert hasattr(m, "delta")
        assert hasattr(m, "runtime_hints")
        assert m.deduplication == {}
        assert m.delta == {}
        assert m.runtime_hints == {}

    def test_block_entry_dedup_ref(self) -> None:
        """BlockEntry has dedup_ref field."""
        b = BlockEntry(
            index=0,
            offset=0,
            compressed_size=10,
            original_size=20,
            codec="raw",
            hash="abc",
        )
        assert b.dedup_ref == -1  # Default

    def test_manifest_v7_serialization(self) -> None:
        """Manifest v7 serializes and deserializes new fields."""
        m = KMCManifest(
            deduplication={"enabled": True, "fingerprint": "sha256"},
            delta={"enabled": False},
            runtime_hints={"partial_file_access": True},
        )
        json_str = m.to_json()
        m2 = KMCManifest.from_json(json_str)
        assert m2.deduplication["enabled"] is True
        assert m2.delta["enabled"] is False
        assert m2.runtime_hints["partial_file_access"] is True

    def test_manifest_v7_backward_compat(self) -> None:
        """Reading a v6 manifest without v7 fields works."""
        v6_data = {
            "version": 6,
            "tool": "kimari-microcompress",
            "tool_version": "0.7.0-alpha",
            "files": [],
        }
        m = KMCManifest.from_json(json.dumps(v6_data))
        assert m.version == 6
        assert m.deduplication == {}
        assert m.delta == {}
        assert m.runtime_hints == {}

    def test_pack_creates_runtime_hints(self, tmp_path: Path) -> None:
        """Pack creates runtime_hints in manifest."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive)

        manifest = inspect(archive)
        assert manifest.runtime_hints.get("partial_file_access") is True
        assert manifest.runtime_hints.get("compressed_inference") is False

    def test_pack_creates_v7_manifest(self, tmp_path: Path) -> None:
        """Pack creates a v7 manifest."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive)

        manifest = inspect(archive)
        assert manifest.version == 7

    def test_block_dedup_ref_serialization(self) -> None:
        """Block dedup_ref serializes and deserializes correctly."""
        m = KMCManifest(
            files=[
                FileEntry(
                    path="test.bin",
                    original_size=100,
                    hash="abc123",
                    block_size=256 * 1024,
                    blocks=[
                        BlockEntry(
                            index=0,
                            offset=16,
                            compressed_size=50,
                            original_size=100,
                            codec="raw",
                            hash="def456",
                            dedup_ref=5,
                        )
                    ],
                )
            ]
        )
        json_str = m.to_json()
        m2 = KMCManifest.from_json(json_str)
        assert m2.files[0].blocks[0].dedup_ref == 5


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


class TestCLIV08:
    """Tests for v0.8 CLI flags."""

    def test_pack_dedup_flag(self, tmp_path: Path) -> None:
        """kmc pack --dedup creates a dedup archive."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        manifest = inspect(archive)
        assert manifest.deduplication.get("enabled") is True

    def test_pack_delta_base_flag(self, tmp_path: Path) -> None:
        """kmc pack --delta-base creates a delta archive."""
        source = _create_test_dir(tmp_path)
        base = tmp_path / "base.kmc"
        pack(source, base)

        delta = tmp_path / "delta.kmc"
        pack(source, delta, delta_base=base)

        manifest = inspect(delta)
        assert manifest.delta.get("enabled") is True

    def test_inspect_shows_dedup(self, tmp_path: Path) -> None:
        """kmc inspect shows dedup info for dedup archives."""
        source = _create_dir_with_duplicates(tmp_path)
        archive = tmp_path / "dedup.kmc"
        pack(source, archive, dedup=True)

        manifest = inspect(archive)
        assert "enabled" in manifest.deduplication

    def test_inspect_shows_runtime_hints(self, tmp_path: Path) -> None:
        """kmc inspect shows runtime hints."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "test.kmc"
        pack(source, archive)

        manifest = inspect(archive)
        assert "partial_file_access" in manifest.runtime_hints

    def test_gguf_aware_codec_path(self, tmp_path: Path) -> None:
        """Pack with gguf_aware uses appropriate codec strategy."""
        source = _create_test_dir(tmp_path)
        archive = tmp_path / "gguf_aware.kmc"
        pack(source, archive, gguf_aware=True)

        report = verify_full(archive)
        assert report.integrity == "OK"
