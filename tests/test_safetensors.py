"""Tests for safetensors format support: metadata, shards, LoRA detection, fallback."""

import json
import struct
from pathlib import Path

from kmc.formats.safetensors import (
    _HAS_SAFETENSORS_LIB,
    _detect_lora_from_tensors,
    _parse_shard_name,
    detect_lora_adapter,
    detect_safetensors_shards,
    get_safetensors_status,
    is_safetensors_available,
    read_safetensors_info,
)


def _make_safetensors_file(
    path: Path,
    tensors: dict | None = None,
    metadata: dict | None = None,
    padding: int = 0,
) -> Path:
    """Create a minimal safetensors file for testing."""
    if tensors is None:
        tensors = {
            "weight1.weight": {
                "dtype": "F32",
                "shape": [64, 64],
                "data_offsets": [0, 16384],
            },
            "weight2.bias": {
                "dtype": "F16",
                "shape": [64],
                "data_offsets": [16384, 17152],
            },
        }

    header_dict = {}
    if metadata:
        header_dict["__metadata__"] = metadata
    header_dict.update(tensors)

    header_json = json.dumps(header_dict).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    data_size = 0
    for t in tensors.values():
        offsets = t.get("data_offsets", [0, 0])
        if len(offsets) >= 2:
            data_size = max(data_size, offsets[1])

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(b"\x00" * (data_size + padding))

    return path


# ---------------------------------------------------------------------------
# read_safetensors_info tests
# ---------------------------------------------------------------------------


class TestReadSafetensorsInfo:
    """Tests for reading safetensors metadata."""

    def test_read_basic_metadata(self, tmp_path: Path):
        """Read basic tensor metadata from a safetensors file."""
        st_file = _make_safetensors_file(tmp_path / "model.safetensors")
        info = read_safetensors_info(st_file)

        assert info.available is True
        assert info.tensor_count == 2
        assert info.total_params == 64 * 64 + 64
        assert "F32" in info.dtypes
        assert "F16" in info.dtypes
        assert len(info.tensors) == 2
        assert info.tensors[0].name == "weight1.weight"
        assert info.tensors[0].dtype == "F32"
        assert info.tensors[0].shape == [64, 64]

    def test_read_with_metadata(self, tmp_path: Path):
        """Read file with __metadata__ field."""
        st_file = _make_safetensors_file(
            tmp_path / "model.safetensors",
            metadata={"model_type": "gpt2", "format": "safetensors"},
        )
        info = read_safetensors_info(st_file)

        assert info.metadata.get("model_type") == "gpt2"

    def test_read_tensor_byte_sizes(self, tmp_path: Path):
        """Verify byte_offset and byte_size are correct."""
        st_file = _make_safetensors_file(tmp_path / "model.safetensors")
        info = read_safetensors_info(st_file)

        # weight1: F32, [64, 64] = 16384 bytes
        assert info.tensors[0].byte_size == 16384
        assert info.tensors[0].byte_offset == 0

        # weight2: F16, [64] = 128 bytes
        assert info.tensors[1].byte_size == 768
        assert info.tensors[1].byte_offset == 16384

    def test_read_file_size(self, tmp_path: Path):
        """File size is recorded correctly."""
        st_file = _make_safetensors_file(tmp_path / "model.safetensors", padding=100)
        info = read_safetensors_info(st_file)

        assert info.file_size > 0
        assert info.file_size == st_file.stat().st_size

    def test_read_nonexistent_file(self, tmp_path: Path):
        """Reading a nonexistent file raises FileNotFoundError."""
        from pytest import raises

        with raises(FileNotFoundError):
            read_safetensors_info(tmp_path / "nonexistent.safetensors")

    def test_read_non_safetensors_file(self, tmp_path: Path):
        """Reading a non-safetensors file raises ValueError."""
        from pytest import raises

        bad_file = tmp_path / "model.bin"
        bad_file.write_bytes(b"\x00" * 100)
        with raises(ValueError, match="[Nn]ot a safetensors"):
            read_safetensors_info(bad_file)

    def test_read_empty_safetensors(self, tmp_path: Path):
        """Read a safetensors file with no tensors (only metadata)."""
        st_file = _make_safetensors_file(
            tmp_path / "model.safetensors",
            tensors={},
            metadata={"model_type": "test"},
        )
        info = read_safetensors_info(st_file)

        assert info.tensor_count == 0
        assert info.total_tensor_bytes == 0
        assert info.total_params == 0


# ---------------------------------------------------------------------------
# Shard detection tests
# ---------------------------------------------------------------------------


class TestShardDetection:
    """Tests for shard name parsing and detection."""

    def test_parse_shard_name_valid(self):
        """Valid shard name is parsed correctly."""
        is_shard, index, total = _parse_shard_name("model-00001-of-00002.safetensors")
        assert is_shard is True
        assert index == 1
        assert total == 2

    def test_parse_shard_name_multi_digit(self):
        """Multi-digit shard numbers are parsed correctly."""
        is_shard, index, total = _parse_shard_name("model-00010-of-00015.safetensors")
        assert is_shard is True
        assert index == 10
        assert total == 15

    def test_parse_shard_name_invalid(self):
        """Non-shard name returns (False, 0, 0)."""
        is_shard, index, total = _parse_shard_name("model.safetensors")
        assert is_shard is False
        assert index == 0
        assert total == 0

    def test_parse_shard_name_bin(self):
        """Shard pattern for .bin files is not matched."""
        is_shard, _, _ = _parse_shard_name("model-00001-of-00002.bin")
        assert is_shard is False

    def test_shard_file_detected_in_info(self, tmp_path: Path):
        """SafetensorsInfo marks is_shard for shard files."""
        shard_file = _make_safetensors_file(tmp_path / "model-00001-of-00003.safetensors")
        info = read_safetensors_info(shard_file)

        assert info.is_shard is True
        assert info.shard_index == 1
        assert info.shard_total == 3

    def test_detect_shards_in_directory(self, tmp_path: Path):
        """detect_safetensors_shards finds shard files."""
        _make_safetensors_file(tmp_path / "model-00001-of-00002.safetensors")
        _make_safetensors_file(tmp_path / "model-00002-of-00002.safetensors")
        # Non-shard file should not be detected
        _make_safetensors_file(tmp_path / "model.safetensors")

        shards = detect_safetensors_shards(tmp_path)
        assert len(shards) == 2
        assert all(f.suffix == ".safetensors" for f in shards)


# ---------------------------------------------------------------------------
# LoRA detection tests
# ---------------------------------------------------------------------------


class TestLoRADetection:
    """Tests for LoRA/PEFT adapter detection."""

    def test_detect_lora_from_tensor_names(self):
        """LoRA is detected from tensor names containing 'lora_A'/'lora_B'."""
        names = ["q_proj.lora_A.weight", "q_proj.lora_B.weight", "k_proj.lora_A.weight"]
        is_lora, rank, modules, base = _detect_lora_from_tensors(names, {})

        assert is_lora is True
        assert len(modules) > 0

    def test_detect_lora_from_metadata(self):
        """LoRA is detected from metadata with base_model_name_or_path."""
        names = ["lora_A.weight", "lora_B.weight"]
        metadata = {"base_model_name_or_path": "meta-llama/Llama-2-7b-hf"}
        is_lora, rank, modules, base = _detect_lora_from_tensors(names, metadata)

        assert is_lora is True
        assert base == "meta-llama/Llama-2-7b-hf"

    def test_no_lora_from_regular_tensors(self):
        """Regular tensors are not misidentified as LoRA."""
        names = ["weight1.weight", "weight2.bias"]
        is_lora, _, _, _ = _detect_lora_from_tensors(names, {})

        assert is_lora is False

    def test_lora_adapter_directory_detection(self, tmp_path: Path):
        """detect_lora_adapter finds adapter_model.safetensors."""
        _make_safetensors_file(
            tmp_path / "adapter_model.safetensors",
            tensors={
                "q_proj.lora_A.weight": {
                    "dtype": "F32",
                    "shape": [8, 64],
                    "data_offsets": [0, 2048],
                },
                "q_proj.lora_B.weight": {
                    "dtype": "F32",
                    "shape": [64, 8],
                    "data_offsets": [2048, 4096],
                },
            },
        )

        result = detect_lora_adapter(tmp_path)
        assert result["is_lora"] is True
        assert result["has_adapter_model"] is True

    def test_lora_with_adapter_config(self, tmp_path: Path):
        """LoRA adapter with adapter_config.json is fully detected."""
        _make_safetensors_file(
            tmp_path / "adapter_model.safetensors",
            tensors={
                "q_proj.lora_A.weight": {
                    "dtype": "F32",
                    "shape": [8, 64],
                    "data_offsets": [0, 2048],
                },
                "q_proj.lora_B.weight": {
                    "dtype": "F32",
                    "shape": [64, 8],
                    "data_offsets": [2048, 4096],
                },
            },
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 8,
                    "target_modules": ["q_proj", "v_proj"],
                    "base_model_name_or_path": "gpt2",
                }
            )
        )

        result = detect_lora_adapter(tmp_path)
        assert result["is_lora"] is True
        assert result["lora_rank"] == 8
        assert "q_proj" in result["target_modules"]
        assert result["base_model_reference"] == "gpt2"

    def test_lora_rank_from_shapes(self, tmp_path: Path):
        """LoRA rank is inferred from tensor shapes."""
        info = read_safetensors_info(
            _make_safetensors_file(
                tmp_path / "adapter_model.safetensors",
                tensors={
                    "q_proj.lora_A.weight": {
                        "dtype": "F32",
                        "shape": [16, 64],
                        "data_offsets": [0, 4096],
                    },
                    "q_proj.lora_B.weight": {
                        "dtype": "F32",
                        "shape": [64, 16],
                        "data_offsets": [4096, 8192],
                    },
                },
            )
        )

        assert info.is_lora is True
        assert info.lora_rank == 16  # From lora_A shape[0]


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for graceful degradation when safetensors is not installed."""

    def test_get_safetensors_status(self):
        """get_safetensors_status returns valid status dict."""
        status = get_safetensors_status()
        assert "status" in status
        assert "reason" in status
        assert "suggestion" in status

        if _HAS_SAFETENSORS_LIB:
            assert status["status"] == "available"
        else:
            assert status["status"] == "unavailable"
            assert "safetensors" in status["suggestion"].lower()

    def test_is_safetensors_available(self):
        """is_safetensors_available returns a boolean."""
        result = is_safetensors_available()
        assert isinstance(result, bool)
        assert result == _HAS_SAFETENSORS_LIB

    def test_pure_python_fallback_works(self, tmp_path: Path):
        """Pure Python fallback parses safetensors header correctly."""
        st_file = _make_safetensors_file(tmp_path / "model.safetensors")
        # This should work regardless of whether the safetensors library is installed
        info = read_safetensors_info(st_file)
        assert info.available is True
        assert info.tensor_count == 2


# ---------------------------------------------------------------------------
# Tensor dtype summary tests
# ---------------------------------------------------------------------------


class TestDtypeSummary:
    """Tests for dtype summary extraction."""

    def test_multiple_dtypes(self, tmp_path: Path):
        """Multiple dtypes are detected and sorted."""
        st_file = _make_safetensors_file(
            tmp_path / "model.safetensors",
            tensors={
                "bf16_weight": {
                    "dtype": "BF16",
                    "shape": [100, 100],
                    "data_offsets": [0, 20000],
                },
                "f32_bias": {
                    "dtype": "F32",
                    "shape": [100],
                    "data_offsets": [20000, 20400],
                },
                "f16_weight": {
                    "dtype": "F16",
                    "shape": [50, 50],
                    "data_offsets": [20400, 20900],
                },
            },
        )
        info = read_safetensors_info(st_file)

        assert info.dtypes == ["BF16", "F16", "F32"]
        assert info.tensor_count == 3
