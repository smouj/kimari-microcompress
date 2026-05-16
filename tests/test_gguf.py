"""Tests for GGUF format parser: magic, version, endianness, tensor metadata, quantization."""

import struct
from pathlib import Path

import pytest

from kmc.formats.gguf import (
    GGUFInfo,
    GGUFTensorInfo,
    is_gguf_file,
    is_quantized_ggml_type,
    read_gguf_info,
)


def _make_gguf_file(
    path: Path,
    version: int = 3,
    tensor_count: int = 10,
    kv_count: int = 5,
    endianness: str = "little",
) -> Path:
    """Create a minimal synthetic GGUF file for testing."""
    GGUF_MAGIC = 0x46475547

    fmt_prefix = "<" if endianness == "little" else ">"

    with open(path, "wb") as f:
        f.write(struct.pack(f"{fmt_prefix}I", GGUF_MAGIC))
        f.write(struct.pack(f"{fmt_prefix}I", version))
        if version >= 2:
            f.write(struct.pack(f"{fmt_prefix}Q", tensor_count))
            f.write(struct.pack(f"{fmt_prefix}Q", kv_count))
        else:
            f.write(struct.pack(f"{fmt_prefix}I", tensor_count))
            f.write(struct.pack(f"{fmt_prefix}I", kv_count))
        f.write(b"\x00" * 256)

    return path


def _make_gguf_file_with_tensors(
    path: Path,
    version: int = 3,
    tensors: list[dict] | None = None,
    kv_pairs: list[tuple] | None = None,
) -> Path:
    """Create a synthetic GGUF file with tensor info section.

    Args:
        tensors: List of dicts with keys: name, shape, type_id, offset
        kv_pairs: List of (key, value_type, value_bytes) tuples
    """
    GGUF_MAGIC = 0x46475547

    if tensors is None:
        tensors = [
            {"name": "token_embd.weight", "shape": [4, 4], "type_id": 0, "offset": 0},
            {"name": "output.weight", "shape": [4, 4], "type_id": 14, "offset": 64},
        ]
    if kv_pairs is None:
        kv_pairs = []

    fmt_prefix = "<"

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack(f"{fmt_prefix}I", GGUF_MAGIC))
        f.write(struct.pack(f"{fmt_prefix}I", version))
        f.write(struct.pack(f"{fmt_prefix}Q", len(tensors)))
        f.write(struct.pack(f"{fmt_prefix}Q", len(kv_pairs)))

        # KV pairs (empty for now)
        for key, vtype, value_bytes in kv_pairs:
            # Key string
            key_bytes = key.encode("utf-8")
            f.write(struct.pack(f"{fmt_prefix}Q", len(key_bytes)))
            f.write(key_bytes)
            # Value type
            f.write(struct.pack(f"{fmt_prefix}I", vtype))
            # Value
            f.write(value_bytes)

        # Tensor info section
        for t in tensors:
            # Name string
            name_bytes = t["name"].encode("utf-8")
            f.write(struct.pack(f"{fmt_prefix}Q", len(name_bytes)))
            f.write(name_bytes)
            # Number of dimensions
            f.write(struct.pack(f"{fmt_prefix}I", len(t["shape"])))
            # Dimensions
            for dim in t["shape"]:
                f.write(struct.pack(f"{fmt_prefix}Q", dim))
            # Type
            f.write(struct.pack(f"{fmt_prefix}I", t["type_id"]))
            # Offset
            f.write(struct.pack(f"{fmt_prefix}Q", t["offset"]))

        # Padding and dummy data
        f.write(b"\x00" * 512)

    return path


class TestReadGGUFInfo:
    """Tests for reading GGUF header information."""

    def test_read_gguf_v3(self, tmp_path: Path):
        """Read a GGUF v3 file."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=3)
        info = read_gguf_info(gguf_file)

        assert info.available is True
        assert info.magic == "GGUF"
        assert info.version == 3
        assert info.endianness == "little"
        assert info.tensor_count == 10
        assert info.metadata_kv_count == 5
        assert info.file_size > 0
        assert info.tensor_metadata_implemented is True

    def test_read_gguf_v2(self, tmp_path: Path):
        """Read a GGUF v2 file."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=2)
        info = read_gguf_info(gguf_file)

        assert info.version == 2
        assert info.tensor_count == 10

    def test_read_gguf_v1(self, tmp_path: Path):
        """Read a GGUF v1 file."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=1)
        info = read_gguf_info(gguf_file)

        assert info.version == 1
        assert info.tensor_count == 10
        assert info.metadata_kv_count == 5

    def test_read_big_endian_gguf(self, tmp_path: Path):
        """Read a big-endian GGUF file."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=3, endianness="big")
        info = read_gguf_info(gguf_file)

        assert info.endianness == "big"
        assert info.version == 3
        assert info.tensor_count == 10

    def test_read_file_size(self, tmp_path: Path):
        """File size is recorded correctly."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=3)
        info = read_gguf_info(gguf_file)

        assert info.file_size == gguf_file.stat().st_size

    def test_read_nonexistent_file(self, tmp_path: Path):
        """Reading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_gguf_info(tmp_path / "nonexistent.gguf")

    def test_read_invalid_gguf_magic(self, tmp_path: Path):
        """File with invalid magic is rejected."""
        bad_file = tmp_path / "bad.gguf"
        with open(bad_file, "wb") as f:
            f.write(b"NOT_GGUF" + b"\x00" * 256)

        with pytest.raises(ValueError, match="[Ii]nvalid GGUF magic"):
            read_gguf_info(bad_file)

    def test_read_truncated_gguf(self, tmp_path: Path):
        """Truncated GGUF file is detected."""
        truncated = tmp_path / "truncated.gguf"
        truncated.write_bytes(struct.pack("<I", 0x46475547))

        with pytest.raises(ValueError, match="[Tt]runcated"):
            read_gguf_info(truncated)

    def test_read_unsupported_version(self, tmp_path: Path):
        """Unsupported GGUF version is rejected."""
        GGUF_MAGIC = 0x46475547
        bad_file = tmp_path / "bad_version.gguf"
        with open(bad_file, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", 99))  # Invalid version
            f.write(b"\x00" * 256)

        with pytest.raises(ValueError, match="[Uu]nsupported"):
            read_gguf_info(bad_file)


class TestGGUFTensorMetadata:
    """Tests for GGUF tensor metadata parsing."""

    def test_parse_tensor_info(self, tmp_path: Path):
        """Parse tensor info from a synthetic GGUF file."""
        gguf_file = _make_gguf_file_with_tensors(tmp_path / "model.gguf")
        info = read_gguf_info(gguf_file, parse_tensors=True)

        assert len(info.tensors) == 2
        assert info.tensors[0].name == "token_embd.weight"
        assert info.tensors[0].shape == [4, 4]
        assert info.tensors[0].ggml_type == "F32"
        assert info.tensors[1].name == "output.weight"
        assert info.tensors[1].ggml_type == "Q6_K"

    def test_quantization_summary(self, tmp_path: Path):
        """Quantization summary is computed correctly."""
        gguf_file = _make_gguf_file_with_tensors(tmp_path / "model.gguf")
        info = read_gguf_info(gguf_file, parse_tensors=True)

        assert "F32" in info.quantization_summary
        assert "Q6_K" in info.quantization_summary
        assert info.quantization_summary["F32"] == 1
        assert info.quantization_summary["Q6_K"] == 1

    def test_tensor_estimated_size(self, tmp_path: Path):
        """Estimated sizes are computed for known types."""
        gguf_file = _make_gguf_file_with_tensors(tmp_path / "model.gguf")
        info = read_gguf_info(gguf_file, parse_tensors=True)

        # F32: 4*4 = 16 elements * 4 bytes = 64 bytes
        f32_tensor = info.tensors[0]
        assert f32_tensor.estimated_size == 64

    def test_parse_tensors_false(self, tmp_path: Path):
        """When parse_tensors=False, tensor list is empty."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf")
        info = read_gguf_info(gguf_file, parse_tensors=False)

        assert info.tensors == []
        assert info.quantization_summary == {}

    def test_unknown_type_id(self, tmp_path: Path):
        """Unknown type IDs are preserved as integers."""
        tensors = [
            {"name": "test.weight", "shape": [4], "type_id": 999, "offset": 0},
        ]
        gguf_file = _make_gguf_file_with_tensors(tmp_path / "model.gguf", tensors=tensors)
        info = read_gguf_info(gguf_file, parse_tensors=True)

        assert len(info.tensors) == 1
        assert info.tensors[0].ggml_type == 999

    def test_v1_tensor_parsing_warning(self, tmp_path: Path):
        """GGUF v1 files get a warning about tensor metadata."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf", version=1)
        info = read_gguf_info(gguf_file, parse_tensors=True)

        assert any("v1" in w.lower() for w in info.warnings)

    def test_partial_parse_with_warnings(self, tmp_path: Path):
        """Partial tensor parsing produces warnings."""
        # Create a file with tensor_count=5 but only write 1 tensor
        GGUF_MAGIC = 0x46475547

        path = tmp_path / "partial.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 5))  # tensor_count=5
            f.write(struct.pack("<Q", 0))  # kv_count=0

            # Write only 1 tensor
            name_bytes = b"test.weight"
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", 1))  # 1 dimension
            f.write(struct.pack("<Q", 4))  # shape=[4]
            f.write(struct.pack("<I", 0))  # F32
            f.write(struct.pack("<Q", 0))  # offset=0

            # No more tensors — file ends
            f.write(b"\x00" * 16)

        info = read_gguf_info(path, parse_tensors=True)

        # Should have at least 1 tensor
        assert len(info.tensors) >= 1
        # Should have warnings about partial parsing
        assert len(info.warnings) > 0


class TestIsGgufFile:
    """Tests for the is_gguf_file detection function."""

    def test_detect_valid_gguf(self, tmp_path: Path):
        """Valid GGUF file is detected."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf")
        assert is_gguf_file(gguf_file) is True

    def test_detect_non_gguf(self, tmp_path: Path):
        """Non-GGUF file is not detected."""
        not_gguf = tmp_path / "model.bin"
        not_gguf.write_bytes(b"\x00" * 256)
        assert is_gguf_file(not_gguf) is False

    def test_detect_nonexistent_file(self, tmp_path: Path):
        """Nonexistent file returns False."""
        assert is_gguf_file(tmp_path / "nonexistent.gguf") is False

    def test_detect_too_small_file(self, tmp_path: Path):
        """File too small for magic returns False."""
        small = tmp_path / "small.gguf"
        small.write_bytes(b"GG")
        assert is_gguf_file(small) is False


class TestGGUFInfoDataclass:
    """Tests for the GGUFInfo dataclass."""

    def test_default_values(self):
        """GGUFInfo has sensible default values."""
        info = GGUFInfo()
        assert info.available is True
        assert info.magic == "GGUF"
        assert info.version == 0
        assert info.endianness == "little"
        assert info.tensor_count == 0
        assert info.tensors == []
        assert info.quantization_summary == {}
        assert info.warnings == []
        assert info.tensor_metadata_implemented is True


class TestGGUFTensorInfoDataclass:
    """Tests for the GGUFTensorInfo dataclass."""

    def test_default_values(self):
        """GGUFTensorInfo has sensible default values."""
        t = GGUFTensorInfo(name="test")
        assert t.name == "test"
        assert t.shape == []
        assert t.ggml_type == "unknown"
        assert t.offset is None
        assert t.estimated_size is None


class TestIsQuantizedGgmlType:
    """Tests for the is_quantized_ggml_type helper."""

    def test_float_types_are_not_quantized(self):
        """F32, F16, BF16 are not quantized."""
        assert is_quantized_ggml_type("F32") is False
        assert is_quantized_ggml_type("F16") is False
        assert is_quantized_ggml_type("BF16") is False

    def test_quantized_types(self):
        """Q4_K, Q6_K, etc. are quantized."""
        assert is_quantized_ggml_type("Q4_K") is True
        assert is_quantized_ggml_type("Q6_K") is True
        assert is_quantized_ggml_type("Q8_0") is True
        assert is_quantized_ggml_type("Q4_0") is True

    def test_integer_type_ids(self):
        """Integer type IDs are handled correctly."""
        assert is_quantized_ggml_type(0) is False  # F32
        assert is_quantized_ggml_type(1) is False  # F16
        assert is_quantized_ggml_type(2) is True  # Q4_0
        assert is_quantized_ggml_type(14) is True  # Q6_K
