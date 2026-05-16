"""Tests for GGUF format parser: magic, version, endianness, synthetic files."""

import struct
from pathlib import Path

import pytest

from kmc.formats.gguf import GGUFInfo, is_gguf_file, read_gguf_info


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
        assert info.tensor_metadata_implemented is False

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

    def test_tensor_metadata_not_implemented(self, tmp_path: Path):
        """tensor_metadata_implemented is always False."""
        gguf_file = _make_gguf_file(tmp_path / "model.gguf")
        info = read_gguf_info(gguf_file)

        assert info.tensor_metadata_implemented is False


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
        assert info.metadata_kv_count == 0
        assert info.tensor_metadata_implemented is False
