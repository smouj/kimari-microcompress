"""Tests for LoRA/PEFT workflows: detection, packing, inspection, roundtrip."""

import json
from pathlib import Path

from kmc.workflows.lora import (
    LoRAAdapterInfo,
    build_lora_manifest_metadata,
    detect_lora_adapter,
)
from kmc.archive import pack, unpack


def _make_lora_adapter(
    tmp_path: Path,
    with_config: bool = True,
    base_model: str | None = None,
) -> Path:
    """Create a synthetic LoRA adapter directory."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    # Create a minimal safetensors file with LoRA tensors
    header = {
        "__metadata__": {},
        "model.layers.0.self_attn.q_proj.lora_A.weight": {
            "dtype": "F16",
            "shape": [16, 4096],
            "data_offsets": [0, 131072],
        },
        "model.layers.0.self_attn.q_proj.lora_B.weight": {
            "dtype": "F16",
            "shape": [4096, 16],
            "data_offsets": [131072, 262144],
        },
    }
    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_len = len(header_json)

    model_path = adapter_dir / "adapter_model.safetensors"
    with open(model_path, "wb") as f:
        f.write(header_len.to_bytes(8, "little"))
        f.write(header_json)
        f.write(b"\x00" * 262144)  # dummy tensor data

    # Create adapter_config.json
    if with_config:
        config = {
            "peft_type": "LORA",
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        }
        if base_model:
            config["base_model_name_or_path"] = base_model
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    # Create README
    (adapter_dir / "README.md").write_text("# LoRA Adapter\n", encoding="utf-8")

    return adapter_dir


class TestDetectLoRAAdapter:
    """Tests for LoRA adapter detection."""

    def test_detect_valid_adapter(self, tmp_path: Path):
        """Detect a valid LoRA adapter directory."""
        adapter_dir = _make_lora_adapter(tmp_path)
        info = detect_lora_adapter(adapter_dir)

        assert info.is_lora is True
        assert info.has_adapter_model is True
        assert info.has_adapter_config is True
        assert info.has_readme is True
        assert info.peft_type == "LORA"
        assert info.lora_rank == 16
        assert "q_proj" in info.target_modules

    def test_adapter_with_base_model(self, tmp_path: Path):
        """Detect base model reference from adapter_config."""
        adapter_dir = _make_lora_adapter(tmp_path, base_model="meta-llama/Llama-2-7b")
        info = detect_lora_adapter(adapter_dir)

        assert info.base_model_name_or_path == "meta-llama/Llama-2-7b"

    def test_adapter_without_config(self, tmp_path: Path):
        """Detect LoRA adapter without adapter_config.json."""
        adapter_dir = _make_lora_adapter(tmp_path, with_config=False)
        info = detect_lora_adapter(adapter_dir)

        assert info.is_lora is True
        assert info.has_adapter_config is False
        assert info.peft_type == "unknown"

    def test_non_lora_directory(self, tmp_path: Path):
        """Non-LoRA directory is not detected as LoRA."""
        empty_dir = tmp_path / "not_lora"
        empty_dir.mkdir()
        (empty_dir / "config.json").write_text("{}", encoding="utf-8")

        info = detect_lora_adapter(empty_dir)
        assert info.is_lora is False

    def test_config_without_model_warning(self, tmp_path: Path):
        """adapter_config.json without adapter model produces warning."""
        adapter_dir = tmp_path / "incomplete_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
        )

        info = detect_lora_adapter(adapter_dir)
        assert info.has_adapter_config is True
        assert info.has_adapter_model is False
        assert len(info.warnings) > 0

    def test_corrupt_config(self, tmp_path: Path):
        """Corrupt adapter_config.json is handled gracefully."""
        adapter_dir = _make_lora_adapter(tmp_path)
        (adapter_dir / "adapter_config.json").write_text("not valid json{{{", encoding="utf-8")

        info = detect_lora_adapter(adapter_dir)
        assert info.is_lora is True  # Still detected from model file
        assert len(info.warnings) > 0


class TestBuildLoraManifestMetadata:
    """Tests for building LoRA manifest metadata."""

    def test_build_metadata(self):
        """Build metadata from a detected LoRA adapter."""
        info = LoRAAdapterInfo(
            is_lora=True,
            peft_type="LORA",
            lora_rank=16,
            target_modules=["q_proj", "v_proj"],
            base_model_name_or_path="meta-llama/Llama-2-7b",
        )
        metadata = build_lora_manifest_metadata(info)

        assert metadata["artifact_type"] == "lora_adapter"
        assert metadata["peft_type"] == "LORA"
        assert metadata["r"] == 16
        assert metadata["target_modules"] == ["q_proj", "v_proj"]
        assert metadata["base_model_name_or_path"] == "meta-llama/Llama-2-7b"

    def test_build_metadata_unknowns(self):
        """Build metadata with unknown values."""
        info = LoRAAdapterInfo(is_lora=True)
        metadata = build_lora_manifest_metadata(info)

        assert metadata["artifact_type"] == "lora_adapter"
        assert metadata["base_model_name_or_path"] == "unknown"
        assert "r" not in metadata  # rank is None


class TestPackLoraRoundtrip:
    """Tests for pack-lora roundtrip."""

    def test_pack_lora_roundtrip(self, tmp_path: Path):
        """Pack a LoRA adapter and unpack it, verify byte-for-byte match."""
        adapter_dir = _make_lora_adapter(tmp_path)
        archive = tmp_path / "adapter.kmc"
        output_dir = tmp_path / "restored"

        from kmc.workflows.lora import build_lora_manifest_metadata

        info = detect_lora_adapter(adapter_dir)

        pack(
            adapter_dir,
            archive,
            tensor_aware=True,
            artifact_type="lora_adapter",
            artifact_metadata=build_lora_manifest_metadata(info),
        )

        assert archive.exists()

        unpack(archive, output_dir)

        # Compare files
        for original_file in adapter_dir.rglob("*"):
            if original_file.is_file():
                rel = original_file.relative_to(adapter_dir)
                restored_file = output_dir / rel
                assert restored_file.exists(), f"Missing: {rel}"
                assert original_file.read_bytes() == restored_file.read_bytes()

    def test_pack_lora_manifest_has_artifact_type(self, tmp_path: Path):
        """Packed LoRA archive manifest has artifact_type='lora_adapter'."""
        adapter_dir = _make_lora_adapter(tmp_path)
        archive = tmp_path / "adapter.kmc"

        from kmc.workflows.lora import build_lora_manifest_metadata

        info = detect_lora_adapter(adapter_dir)

        pack(
            adapter_dir,
            archive,
            tensor_aware=True,
            artifact_type="lora_adapter",
            artifact_metadata=build_lora_manifest_metadata(info),
        )

        from kmc.archive import inspect as inspect_archive

        manifest = inspect_archive(archive)
        assert manifest.artifact_type == "lora_adapter"
        assert manifest.artifact_metadata.get("peft_type") == "LORA"
