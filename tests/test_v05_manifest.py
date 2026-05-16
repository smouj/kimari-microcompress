"""Tests for v0.5 manifest: artifact_type, artifact_metadata, format_metadata."""

import json

from kmc.manifest import (
    KMCManifest,
    KMC_MANIFEST_VERSION,
)


class TestManifestV4:
    """Tests for manifest v4 (v0.5) features."""

    def test_manifest_version_is_4(self):
        """KMC_MANIFEST_VERSION is 4 for v0.5."""
        assert KMC_MANIFEST_VERSION == 4

    def test_default_artifact_type(self):
        """Default artifact_type is 'unknown'."""
        m = KMCManifest()
        assert m.artifact_type == "unknown"

    def test_default_artifact_metadata(self):
        """Default artifact_metadata is empty dict."""
        m = KMCManifest()
        assert m.artifact_metadata == {}

    def test_default_format_metadata(self):
        """Default format_metadata is empty dict."""
        m = KMCManifest()
        assert m.format_metadata == {}

    def test_tool_version_is_0_5(self):
        """Tool version is 0.5.0-alpha."""
        m = KMCManifest()
        assert m.tool_version == "0.5.0-alpha"

    def test_set_artifact_type(self):
        """artifact_type can be set."""
        m = KMCManifest(artifact_type="lora_adapter")
        assert m.artifact_type == "lora_adapter"

    def test_set_artifact_metadata(self):
        """artifact_metadata can be set."""
        m = KMCManifest(
            artifact_type="lora_adapter",
            artifact_metadata={"peft_type": "LORA", "r": 16},
        )
        assert m.artifact_metadata["peft_type"] == "LORA"
        assert m.artifact_metadata["r"] == 16

    def test_set_format_metadata_gguf(self):
        """format_metadata can contain GGUF info."""
        m = KMCManifest(
            format_metadata={
                "gguf": {
                    "version": 3,
                    "tensor_count": 291,
                    "quantization_summary": {"Q4_K_M": 180, "F16": 71},
                }
            }
        )
        assert "gguf" in m.format_metadata
        assert m.format_metadata["gguf"]["version"] == 3

    def test_set_format_metadata_safetensors(self):
        """format_metadata can contain safetensors info."""
        m = KMCManifest(
            format_metadata={
                "safetensors": {
                    "is_sharded": False,
                    "tensor_count": 100,
                    "dtypes": ["F16", "BF16"],
                }
            }
        )
        assert "safetensors" in m.format_metadata


class TestManifestV4Serialization:
    """Tests for v4 manifest JSON serialization."""

    def test_to_json_includes_artifact_type(self):
        """Serialized JSON includes artifact_type."""
        m = KMCManifest(artifact_type="gguf_model")
        j = m.to_json()
        data = json.loads(j)
        assert data["artifact_type"] == "gguf_model"

    def test_to_json_includes_artifact_metadata(self):
        """Serialized JSON includes artifact_metadata."""
        m = KMCManifest(
            artifact_type="training_checkpoint",
            artifact_metadata={"step": 1000},
        )
        j = m.to_json()
        data = json.loads(j)
        assert data["artifact_metadata"]["step"] == 1000

    def test_to_json_includes_format_metadata(self):
        """Serialized JSON includes format_metadata."""
        m = KMCManifest(format_metadata={"gguf": {"version": 3}})
        j = m.to_json()
        data = json.loads(j)
        assert data["format_metadata"]["gguf"]["version"] == 3

    def test_roundtrip_json(self):
        """JSON roundtrip preserves all v4 fields."""
        m = KMCManifest(
            artifact_type="lora_adapter",
            artifact_metadata={"peft_type": "LORA", "r": 16},
            format_metadata={"safetensors": {"tensor_count": 50}},
        )
        j = m.to_json()
        m2 = KMCManifest.from_json(j)

        assert m2.artifact_type == "lora_adapter"
        assert m2.artifact_metadata["peft_type"] == "LORA"
        assert m2.artifact_metadata["r"] == 16
        assert m2.format_metadata["safetensors"]["tensor_count"] == 50

    def test_roundtrip_bytes(self):
        """Bytes roundtrip preserves all v4 fields."""
        m = KMCManifest(
            artifact_type="training_checkpoint",
            artifact_metadata={"step": 2000},
        )
        b = m.to_bytes()
        m2 = KMCManifest.from_bytes(b)

        assert m2.artifact_type == "training_checkpoint"
        assert m2.artifact_metadata["step"] == 2000


class TestManifestBackwardCompat:
    """Tests for backward compatibility with v1/v2/v3 manifests."""

    def test_read_v1_manifest(self):
        """v1 manifest (no artifact fields) reads with defaults."""
        v1_json = json.dumps(
            {
                "version": 1,
                "tool": "kimari-microcompress",
                "tool_version": "0.1.0",
                "created_at": "2024-01-01",
                "total_original_size": 1000,
                "total_compressed_size": 500,
                "files": [
                    {
                        "path": "test.bin",
                        "original_size": 1000,
                        "hash": "abc123",
                        "block_size": 262144,
                        "blocks": [
                            {
                                "index": 0,
                                "offset": 100,
                                "compressed_size": 500,
                                "original_size": 1000,
                                "codec": "zstd",
                                "hash": "def456",
                            }
                        ],
                    }
                ],
            }
        )
        m = KMCManifest.from_json(v1_json)

        assert m.version == 1
        assert m.artifact_type == "unknown"
        assert m.artifact_metadata == {}
        assert m.format_metadata == {}
        assert len(m.files) == 1
        assert m.files[0].path == "test.bin"

    def test_read_v2_manifest(self):
        """v2 manifest (with tensor entries) reads correctly."""
        v2_json = json.dumps(
            {
                "version": 2,
                "tool": "kimari-microcompress",
                "tool_version": "0.3.0-alpha",
                "created_at": "2024-06-01",
                "total_original_size": 2000,
                "total_compressed_size": 1000,
                "files": [
                    {
                        "path": "model.safetensors",
                        "original_size": 2000,
                        "hash": "abc123",
                        "block_size": 262144,
                        "blocks": [],
                        "tensor_count": 5,
                        "dtype_summary": ["F16"],
                        "tensor_entries": [
                            {
                                "name": "weight",
                                "dtype": "F16",
                                "shape": [4, 4],
                                "byte_offset": 0,
                                "byte_size": 32,
                            }
                        ],
                    }
                ],
            }
        )
        m = KMCManifest.from_json(v2_json)

        assert m.version == 2
        assert m.artifact_type == "unknown"
        assert m.files[0].tensor_count == 5
        assert len(m.files[0].tensor_entries) == 1

    def test_read_v3_manifest(self):
        """v3 manifest (with codec_metadata) reads correctly."""
        v3_json = json.dumps(
            {
                "version": 3,
                "tool": "kimari-microcompress",
                "tool_version": "0.4.0-alpha",
                "created_at": "2024-09-01",
                "total_original_size": 3000,
                "total_compressed_size": 1500,
                "files": [
                    {
                        "path": "model.safetensors",
                        "original_size": 3000,
                        "hash": "abc123",
                        "block_size": 262144,
                        "blocks": [
                            {
                                "index": 0,
                                "offset": 100,
                                "compressed_size": 1500,
                                "original_size": 3000,
                                "codec": "byteplane",
                                "hash": "def456",
                                "codec_metadata": {
                                    "transform": "reorder",
                                    "element_size": 2,
                                },
                                "tensor_name": "weight",
                                "tensor_dtype": "BF16",
                                "tensor_shape": [4, 4],
                            }
                        ],
                    }
                ],
            }
        )
        m = KMCManifest.from_json(v3_json)

        assert m.version == 3
        assert m.artifact_type == "unknown"
        block = m.files[0].blocks[0]
        assert block.codec == "byteplane"
        assert block.codec_metadata["transform"] == "reorder"
        assert block.tensor_dtype == "BF16"

    def test_v4_manifest_with_all_fields(self):
        """v4 manifest with all new fields reads correctly."""
        v4_json = json.dumps(
            {
                "version": 4,
                "tool": "kimari-microcompress",
                "tool_version": "0.5.0-alpha",
                "created_at": "2025-01-01",
                "total_original_size": 5000,
                "total_compressed_size": 2500,
                "artifact_type": "lora_adapter",
                "artifact_metadata": {"peft_type": "LORA", "r": 8},
                "format_metadata": {"safetensors": {"tensor_count": 10}},
                "files": [],
            }
        )
        m = KMCManifest.from_json(v4_json)

        assert m.version == 4
        assert m.artifact_type == "lora_adapter"
        assert m.artifact_metadata["r"] == 8
        assert m.format_metadata["safetensors"]["tensor_count"] == 10
