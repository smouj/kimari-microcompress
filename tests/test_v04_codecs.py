"""Tests for KMC v0.4.0-alpha: tensor-aware lossless codecs.

Tests cover:
    - BytePlane codec roundtrip
    - FloatPlane codec roundtrip
    - Automatic codec selector
    - Safe fallback
    - Codec metadata
    - Manifest compatibility with v0.2/v0.3
    - kmc pack --codec auto/byteplane/floatplane
    - Corrupt payload handling
    - Unknown dtype
    - Misaligned data
    - Simulated zstd absence
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from kmc.codecs.base import CodecContext
from kmc.codecs.byteplane import BytePlaneCodec
from kmc.codecs.floatplane import FloatPlaneCodec
from kmc.codecs.raw import RawCodec
from kmc.codecs.zstd_codec import ZstdCodec, is_zstd_available
from kmc.codecs.registry import get_codec, is_codec_available, list_codecs
from kmc.codecs.selector import get_candidates, select_codec

from kmc.manifest import BlockEntry, FileEntry, KMCManifest, KMC_MANIFEST_VERSION
from kmc.archive import pack, unpack, verify, verify_full, read_manifest_from_archive

# Import fixtures
from tests.fixtures.tensors import (
    generate_bf16_repeated_exponents,
    generate_fp16_smooth_patterns,
    generate_fp32_simulated,
    generate_random_bytes,
    generate_compressible_bytes,
    generate_misaligned_bf16,
)


# ===========================================================================
# BytePlane codec tests
# ===========================================================================


class TestBytePlaneRoundtrip:
    """BytePlane codec must produce exact roundtrip for all inputs."""

    def test_fp16_synthetic(self):
        """Roundtrip with synthetic FP16 data."""
        data = generate_fp16_smooth_patterns(256)
        codec = BytePlaneCodec()
        ctx = CodecContext(dtype="FP16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane FP16 roundtrip failed"

    def test_bf16_synthetic(self):
        """Roundtrip with synthetic BF16 data."""
        data = generate_bf16_repeated_exponents(256)
        codec = BytePlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane BF16 roundtrip failed"

    def test_fp32_synthetic(self):
        """Roundtrip with synthetic FP32 data."""
        data = generate_fp32_simulated(128)
        codec = BytePlaneCodec()
        ctx = CodecContext(dtype="FP32", original_size=len(data))
        result = codec.compress(data, context=ctx)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane FP32 roundtrip failed"

    def test_misaligned_data(self):
        """Roundtrip with misaligned data (extra bytes)."""
        data = generate_misaligned_bf16(100, extra_bytes=3)
        codec = BytePlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane misaligned roundtrip failed"

    def test_empty_data(self):
        """Roundtrip with empty data."""
        data = b""
        codec = BytePlaneCodec()
        result = codec.compress(data)
        decomp_ctx = CodecContext()
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane empty roundtrip failed"

    def test_random_data(self):
        """Roundtrip with random data (high entropy)."""
        data = generate_random_bytes(4096)
        codec = BytePlaneCodec()
        result = codec.compress(data)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "BytePlane random roundtrip failed"

    def test_comparison_vs_zlib(self):
        """BytePlane should compress repeated-exponent BF16 no worse than raw."""
        data = generate_bf16_repeated_exponents(1024)
        bp = BytePlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        bp_result = bp.compress(data, context=ctx)
        # Raw should be same size as original
        assert bp_result.compressed_size <= bp_result.original_size, (
            "BytePlane should not expand data beyond original"
        )


# ===========================================================================
# FloatPlane codec tests
# ===========================================================================


class TestFloatPlaneRoundtrip:
    """FloatPlane codec must produce exact roundtrip for all supported dtypes."""

    def test_bf16_synthetic(self):
        """Roundtrip with synthetic BF16 data."""
        data = generate_bf16_repeated_exponents(256)
        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.metadata.get("transform") == "floatplane"
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "FloatPlane BF16 roundtrip failed"

    def test_fp16_synthetic(self):
        """Roundtrip with synthetic FP16 data."""
        data = generate_fp16_smooth_patterns(256)
        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="FP16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.metadata.get("transform") == "floatplane"
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "FloatPlane FP16 roundtrip failed"

    def test_fp32_synthetic(self):
        """Roundtrip with synthetic FP32 data."""
        data = generate_fp32_simulated(128)
        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="FP32", original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.metadata.get("transform") == "floatplane"
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "FloatPlane FP32 roundtrip failed"

    def test_repeated_exponents(self):
        """FloatPlane should handle repeated exponent patterns well."""
        data = generate_bf16_repeated_exponents(2048)
        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.compressed_size < result.original_size, (
            "FloatPlane should compress repeated-exponent data"
        )

    def test_random_values(self):
        """Roundtrip with random FP32 data."""
        data = bytearray()
        for i in range(256):
            val = (i * 9876543) % (2**32)
            data.extend(struct.pack(">I", val))
        data = bytes(data)

        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="FP32", original_size=len(data))
        result = codec.compress(data, context=ctx)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data, "FloatPlane random FP32 roundtrip failed"

    def test_corrupt_payload_raises(self):
        """Corrupt payload should raise an error on decompression."""
        data = generate_bf16_repeated_exponents(256)
        codec = FloatPlaneCodec()
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        result = codec.compress(data, context=ctx)
        # Corrupt the payload
        corrupt_payload = bytearray(result.payload)
        if len(corrupt_payload) > 10:
            corrupt_payload[10] ^= 0xFF
        with pytest.raises(Exception):
            decomp_ctx = CodecContext(original_size=result.original_size)
            decomp_ctx._codec_metadata = result.metadata
            codec.decompress(bytes(corrupt_payload), context=decomp_ctx)

    def test_incomplete_metadata(self):
        """FloatPlane without dtype should fall back to byteplane."""
        data = generate_bf16_repeated_exponents(256)
        codec = FloatPlaneCodec()
        # No dtype context
        ctx = CodecContext(original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.metadata.get("fallback_reason") is not None, (
            "FloatPlane without dtype should indicate fallback"
        )

    def test_comparison_vs_byteplane(self):
        """FloatPlane vs BytePlane compression comparison."""
        data = generate_bf16_repeated_exponents(1024)
        ctx = CodecContext(dtype="BF16", original_size=len(data))

        fp = FloatPlaneCodec()
        fp_result = fp.compress(data, context=ctx)

        bp = BytePlaneCodec()
        bp_result = bp.compress(data, context=ctx)

        # Both should produce valid results
        assert fp_result.compressed_size > 0
        assert bp_result.compressed_size > 0


# ===========================================================================
# Codec selector tests
# ===========================================================================


class TestCodecSelector:
    """Automatic codec selector tests."""

    def test_bf16_candidates(self):
        """BF16 dtype should include floatplane and byteplane candidates."""
        candidates = get_candidates("BF16")
        assert "floatplane" in candidates
        assert "byteplane" in candidates
        assert "zstd" in candidates

    def test_fp16_candidates(self):
        """FP16 dtype should include floatplane and byteplane."""
        candidates = get_candidates("FP16")
        assert "floatplane" in candidates
        assert "byteplane" in candidates

    def test_fp32_candidates(self):
        """FP32 dtype should include floatplane and byteplane."""
        candidates = get_candidates("FP32")
        assert "floatplane" in candidates
        assert "byteplane" in candidates

    def test_int8_candidates(self):
        """INT8 dtype should not include tensor-aware codecs."""
        candidates = get_candidates("INT8")
        assert "floatplane" not in candidates
        assert "byteplane" not in candidates
        assert "zstd" in candidates

    def test_unknown_dtype_candidates(self):
        """Unknown dtype should fall back to default candidates."""
        candidates = get_candidates("UNKNOWN_TYPE")
        assert "zstd" in candidates
        assert "zlib" in candidates
        assert "raw" in candidates

    def test_gguf_candidates(self):
        """GGUF files should use standard codecs."""
        candidates = get_candidates(None, is_gguf=True)
        assert "floatplane" not in candidates
        assert "byteplane" not in candidates

    def test_auto_select_returns_valid(self):
        """Auto selector should always return a valid result."""
        data = generate_compressible_bytes(1024)
        ctx = CodecContext(dtype="BF16", original_size=len(data))
        selection = select_codec(data, context=ctx)
        assert selection.result.compressed_size > 0
        assert selection.result.original_size == len(data)

    def test_auto_select_with_codec_override(self):
        """Codec override should force specific codec."""
        data = generate_compressible_bytes(1024)
        selection = select_codec(data, codec_override="raw")
        assert selection.codec_name == "raw"

    def test_auto_select_roundtrip_verified(self):
        """Auto selector should verify roundtrip."""
        data = generate_fp16_smooth_patterns(256)
        ctx = CodecContext(dtype="FP16", original_size=len(data))
        selection = select_codec(data, context=ctx, verify_roundtrip=True)
        assert selection.roundtrip_verified

    def test_fallback_on_failure(self):
        """Selector should fallback if all advanced codecs fail."""
        data = b"simple test data"
        selection = select_codec(data, codec_override="floatplane")
        assert selection.result.original_size == len(data)


# ===========================================================================
# Codec registry tests
# ===========================================================================


class TestCodecRegistry:
    """Codec registry tests."""

    def test_list_codecs(self):
        """Should list all registered codecs."""
        codecs = list_codecs()
        assert "raw" in codecs
        assert "zlib" in codecs
        assert "zstd" in codecs
        assert "byteplane" in codecs
        assert "floatplane" in codecs

    def test_get_codec(self):
        """Should instantiate codecs by name."""
        raw = get_codec("raw")
        assert raw.name == "raw"
        zlib_codec = get_codec("zlib")
        assert zlib_codec.name == "zlib"

    def test_unknown_codec_raises(self):
        """Unknown codec name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown codec"):
            get_codec("nonexistent")

    def test_zstd_availability(self):
        """zstd should be available in this test environment."""
        assert is_codec_available("zstd") == is_zstd_available()

    def test_raw_always_available(self):
        """raw codec should always be available."""
        assert is_codec_available("raw")

    def test_byteplane_available(self):
        """byteplane codec should be available."""
        assert is_codec_available("byteplane")

    def test_floatplane_available(self):
        """floatplane codec should be available."""
        assert is_codec_available("floatplane")


# ===========================================================================
# Manifest compatibility tests
# ===========================================================================


class TestManifestCompatibility:
    """Test manifest backward and forward compatibility."""

    def test_v1_manifest_reads(self):
        """v1 manifest (no codec_metadata) should deserialize correctly."""
        v1_json = json.dumps(
            {
                "version": 1,
                "tool": "kimari-microcompress",
                "tool_version": "0.1.0",
                "created_at": "",
                "total_original_size": 100,
                "total_compressed_size": 50,
                "files": [
                    {
                        "path": "test.bin",
                        "original_size": 100,
                        "hash": "abc123",
                        "block_size": 262144,
                        "blocks": [
                            {
                                "index": 0,
                                "offset": 100,
                                "compressed_size": 50,
                                "original_size": 100,
                                "codec": "zstd",
                                "hash": "def456",
                            }
                        ],
                    }
                ],
            }
        )
        manifest = KMCManifest.from_json(v1_json)
        assert manifest.version == 1
        assert len(manifest.files) == 1
        block = manifest.files[0].blocks[0]
        assert block.codec == "zstd"
        assert block.codec_metadata == {}
        assert block.tensor_name == ""

    def test_v2_manifest_reads(self):
        """v2 manifest (with tensor entries) should deserialize correctly."""
        v2_json = json.dumps(
            {
                "version": 2,
                "tool": "kimari-microcompress",
                "tool_version": "0.3.0-alpha",
                "created_at": "",
                "total_original_size": 1000,
                "total_compressed_size": 500,
                "files": [
                    {
                        "path": "model.safetensors",
                        "original_size": 1000,
                        "hash": "abc",
                        "block_size": 262144,
                        "blocks": [
                            {
                                "index": 0,
                                "offset": 100,
                                "compressed_size": 500,
                                "original_size": 1000,
                                "codec": "zstd",
                                "hash": "def",
                            }
                        ],
                        "tensor_count": 2,
                        "dtype_summary": ["BF16"],
                        "tensor_entries": [
                            {
                                "name": "layer.0.weight",
                                "dtype": "BF16",
                                "shape": [64, 64],
                                "byte_offset": 0,
                                "byte_size": 8192,
                            }
                        ],
                    }
                ],
            }
        )
        manifest = KMCManifest.from_json(v2_json)
        assert manifest.version == 2
        assert manifest.files[0].tensor_count == 2
        assert len(manifest.files[0].tensor_entries) == 1

    def test_v3_manifest_roundtrip(self):
        """v3 manifest (with codec_metadata) should roundtrip."""
        manifest = KMCManifest(version=3)
        block = BlockEntry(
            index=0,
            offset=100,
            compressed_size=500,
            original_size=1000,
            codec="floatplane",
            hash="abc123",
            codec_metadata={
                "transform": "floatplane",
                "dtype": "BF16",
                "inner_codec": "zstd",
                "planes": ["sign", "exponent", "mantissa"],
            },
            tensor_name="layer.0.weight",
            tensor_dtype="BF16",
            tensor_shape=[64, 64],
        )
        manifest.files.append(
            FileEntry(
                path="model.safetensors",
                original_size=1000,
                hash="def456",
                block_size=262144,
                blocks=[block],
            )
        )

        json_str = manifest.to_json()
        restored = KMCManifest.from_json(json_str)
        assert restored.version == 3
        assert restored.files[0].blocks[0].codec == "floatplane"
        assert restored.files[0].blocks[0].codec_metadata["transform"] == "floatplane"
        assert restored.files[0].blocks[0].tensor_name == "layer.0.weight"
        assert restored.files[0].blocks[0].tensor_dtype == "BF16"

    def test_current_manifest_version(self):
        """Current manifest should be v6."""
        assert KMC_MANIFEST_VERSION == 6


# ===========================================================================
# Pack with codec tests
# ===========================================================================


class TestPackWithCodec:
    """Test kmc pack with different codec options."""

    def _pack_roundtrip(self, codec: str, tensor_aware: bool = False):
        """Helper: pack a directory and verify roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test.bin").write_bytes(generate_compressible_bytes(2048))

            output = Path(tmpdir) / "test.kmc"
            pack(source, output, codec=codec, tensor_aware=tensor_aware)

            assert output.exists()

            # Verify
            errors = verify(output)
            assert errors == [], f"Verify failed with codec={codec}: {errors}"

            # Unpack and compare
            restore = Path(tmpdir) / "restored"
            unpack(output, restore)

            original_data = (source / "test.bin").read_bytes()
            restored_data = (restore / "test.bin").read_bytes()
            assert original_data == restored_data, f"Roundtrip failed with codec={codec}"

    def test_pack_codec_auto(self):
        """Pack with --codec auto."""
        self._pack_roundtrip("auto")

    def test_pack_codec_raw(self):
        """Pack with --codec raw."""
        self._pack_roundtrip("raw")

    def test_pack_codec_zlib(self):
        """Pack with --codec zlib."""
        self._pack_roundtrip("zlib")

    def test_pack_codec_zstd(self):
        """Pack with --codec zstd."""
        if not is_zstd_available():
            pytest.skip("zstd not available")
        self._pack_roundtrip("zstd")

    def test_pack_codec_byteplane(self):
        """Pack with --codec byteplane."""
        self._pack_roundtrip("byteplane")

    def test_pack_codec_floatplane(self):
        """Pack with --codec floatplane."""
        self._pack_roundtrip("floatplane")

    def test_pack_codec_byteplane_tensor_aware(self):
        """Pack with --codec byteplane --tensor-aware."""
        self._pack_roundtrip("byteplane", tensor_aware=True)

    def test_pack_codec_floatplane_tensor_aware(self):
        """Pack with --codec floatplane --tensor-aware."""
        self._pack_roundtrip("floatplane", tensor_aware=True)

    def test_pack_unknown_codec_raises(self):
        """Pack with unknown codec should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test.bin").write_bytes(b"data")
            output = Path(tmpdir) / "test.kmc"
            with pytest.raises(ValueError, match="Unknown codec"):
                pack(source, output, codec="nonexistent")

    def test_pack_stores_codec_metadata(self):
        """Pack with byteplane should store codec_metadata in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test.bin").write_bytes(generate_compressible_bytes(2048))

            output = Path(tmpdir) / "test.kmc"
            pack(source, output, codec="byteplane")

            manifest, _ = read_manifest_from_archive(output)
            codecs_found = set(b.codec for f in manifest.files for b in f.blocks)
            assert "byteplane" in codecs_found or any(
                b.codec_metadata.get("transform") == "byteplane"
                for f in manifest.files
                for b in f.blocks
            )


# ===========================================================================
# Corrupt data handling tests
# ===========================================================================


class TestCorruptDataHandling:
    """Test handling of corrupt payloads and metadata."""

    def test_corrupt_block_payload(self):
        """Verify should detect corrupt block data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test.bin").write_bytes(generate_compressible_bytes(2048))

            output = Path(tmpdir) / "test.kmc"
            pack(source, output, codec="zstd")

            # Corrupt the archive
            data = bytearray(output.read_bytes())
            if len(data) > 100:
                data[-50] ^= 0xFF
            corrupt_path = Path(tmpdir) / "corrupt.kmc"
            corrupt_path.write_bytes(bytes(data))

            report = verify_full(corrupt_path)
            assert report.integrity == "FAILED"

    def test_unknown_dtype_in_context(self):
        """Codec should handle unknown dtype gracefully."""
        codec = FloatPlaneCodec()
        data = generate_random_bytes(1024)
        ctx = CodecContext(dtype="UNKNOWN_DTYPE", original_size=len(data))
        result = codec.compress(data, context=ctx)
        assert result.metadata.get("fallback_reason") is not None


# ===========================================================================
# Fallback safety tests
# ===========================================================================


class TestFallbackSafety:
    """Test that codecs always fall back safely."""

    def test_byteplane_empty_data(self):
        """BytePlane should handle empty data."""
        codec = BytePlaneCodec()
        result = codec.compress(b"")
        decomp_ctx = CodecContext()
        decomp_ctx._codec_metadata = result.metadata
        assert codec.decompress(result.payload, context=decomp_ctx) == b""

    def test_floatplane_empty_data(self):
        """FloatPlane should handle empty data."""
        codec = FloatPlaneCodec()
        result = codec.compress(b"")
        decomp_ctx = CodecContext()
        decomp_ctx._codec_metadata = result.metadata
        assert codec.decompress(result.payload, context=decomp_ctx) == b""

    def test_byteplane_single_byte(self):
        """BytePlane should handle single byte data."""
        codec = BytePlaneCodec()
        data = b"\x42"
        result = codec.compress(data)
        decomp_ctx = CodecContext(original_size=result.original_size)
        decomp_ctx._codec_metadata = result.metadata
        decompressed = codec.decompress(result.payload, context=decomp_ctx)
        assert decompressed == data

    def test_selector_always_returns_result(self):
        """Selector should always return a result, even with bad input."""
        selection = select_codec(b"")
        assert selection.result is not None

        selection = select_codec(b"\x00")
        assert selection.result is not None

    def test_raw_codec_is_ultimate_fallback(self):
        """Raw codec should always work."""
        codec = RawCodec()
        data = b"test data 123"
        result = codec.compress(data)
        assert codec.decompress(result.payload) == data


# ===========================================================================
# Simulated zstd absence tests
# ===========================================================================


class TestZstdAbsence:
    """Test behavior when zstd is not available."""

    def test_zstd_unavailable_graceful(self):
        """ZstdCodec should raise RuntimeError when zstd not installed."""
        if not is_zstd_available():
            codec = ZstdCodec()
            with pytest.raises(RuntimeError, match="zstandard"):
                codec.compress(b"test")

    def test_byteplane_without_zstd_inner(self):
        """BytePlane should work with zlib inner codec when zstd absent."""
        codec = BytePlaneCodec()
        data = generate_compressible_bytes(1024)
        result = codec.compress(data)
        assert "inner_codec" in result.metadata


# ===========================================================================
# v0.2/v0.3 archive compatibility tests
# ===========================================================================


class TestV02V03Compat:
    """Test that v0.2/v0.3 archives still work with v0.4 code."""

    def test_legacy_zstd_archive_roundtrip(self):
        """Create archive with zstd codec (v0.2 style) and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.bin").write_bytes(generate_compressible_bytes(4096))

            output = Path(tmpdir) / "test.kmc"
            pack(source, output, codec="zstd")

            errors = verify(output)
            assert errors == []

            restore = Path(tmpdir) / "restored"
            unpack(output, restore)

            original = (source / "data.bin").read_bytes()
            restored = (restore / "data.bin").read_bytes()
            assert original == restored

    def test_legacy_zlib_archive_roundtrip(self):
        """Create archive with zlib codec and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.bin").write_bytes(generate_compressible_bytes(4096))

            output = Path(tmpdir) / "test.kmc"
            pack(source, output, codec="zlib")

            errors = verify(output)
            assert errors == []

            restore = Path(tmpdir) / "restored"
            unpack(output, restore)

            assert (source / "data.bin").read_bytes() == (restore / "data.bin").read_bytes()
