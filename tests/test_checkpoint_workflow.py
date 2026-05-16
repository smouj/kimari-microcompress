"""Tests for checkpoint workflows: detection, packing, inspection, roundtrip."""

import json
from pathlib import Path

from kmc.workflows.checkpoint import (
    CheckpointInfo,
    build_checkpoint_manifest_metadata,
    detect_checkpoint,
)
from kmc.archive import pack, unpack


def _make_checkpoint(
    tmp_path: Path,
    step: int = 1000,
    with_safetensors: bool = True,
) -> Path:
    """Create a synthetic training checkpoint directory."""
    ckpt_dir = tmp_path / f"checkpoint-{step}"
    ckpt_dir.mkdir()

    # Create trainer_state.json
    trainer_state = {
        "global_step": step,
        "epoch": step / 100,
        "best_metric": 0.5,
    }
    (ckpt_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")

    # Create global_step.json
    (ckpt_dir / "global_step.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")

    # Create training_args.bin (dummy, not pickle)
    (ckpt_dir / "training_args.bin").write_bytes(b"\x00" * 1024)

    # Create optimizer.pt (dummy, not pickle)
    (ckpt_dir / "optimizer.pt").write_bytes(b"\x00" * 2048)

    # Create scheduler.pt (dummy)
    (ckpt_dir / "scheduler.pt").write_bytes(b"\x00" * 512)

    # Create rng_state.pth (dummy)
    (ckpt_dir / "rng_state.pth").write_bytes(b"\x00" * 256)

    if with_safetensors:
        # Create model.safetensors with minimal content
        header = {
            "__metadata__": {},
            "model.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, 64],
            },
        }
        header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
        header_len = len(header_json)
        with open(ckpt_dir / "model.safetensors", "wb") as f:
            f.write(header_len.to_bytes(8, "little"))
            f.write(header_json)
            f.write(b"\x00" * 64)

    # Create config.json
    (ckpt_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")

    return ckpt_dir


class TestDetectCheckpoint:
    """Tests for checkpoint detection."""

    def test_detect_valid_checkpoint(self, tmp_path: Path):
        """Detect a valid training checkpoint directory."""
        ckpt_dir = _make_checkpoint(tmp_path)
        info = detect_checkpoint(ckpt_dir)

        assert info.is_checkpoint is True
        assert info.step == 1000
        assert info.has_trainer_state is True
        assert info.has_optimizer_state is True
        assert info.has_scheduler_state is True
        assert info.has_rng_state is True
        assert info.has_safetensors_model is True

    def test_detect_step_from_directory_name(self, tmp_path: Path):
        """Step number is inferred from directory name."""
        ckpt_dir = _make_checkpoint(tmp_path, step=2000)
        info = detect_checkpoint(ckpt_dir)
        assert info.step == 2000

    def test_detect_step_from_global_step_json(self, tmp_path: Path):
        """Step number is read from global_step.json."""
        ckpt_dir = tmp_path / "my_checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "global_step.json").write_text(
            json.dumps({"global_step": 500}), encoding="utf-8"
        )
        (ckpt_dir / "model.safetensors").write_bytes(b"\x00" * 128)

        info = detect_checkpoint(ckpt_dir)
        assert info.step == 500

    def test_detect_step_from_trainer_state(self, tmp_path: Path):
        """Step number is read from trainer_state.json."""
        ckpt_dir = tmp_path / "my_checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps({"global_step": 300}), encoding="utf-8"
        )

        info = detect_checkpoint(ckpt_dir)
        assert info.step == 300

    def test_has_pytorch_model_warning(self, tmp_path: Path):
        """pytorch_model.bin presence triggers a warning."""
        ckpt_dir = _make_checkpoint(tmp_path, with_safetensors=False)
        (ckpt_dir / "pytorch_model.bin").write_bytes(b"\x00" * 512)

        info = detect_checkpoint(ckpt_dir)
        assert info.has_pytorch_model is True
        assert any("pickle" in w.lower() for w in info.warnings)

    def test_optimizer_warning(self, tmp_path: Path):
        """optimizer.pt presence triggers a warning."""
        ckpt_dir = _make_checkpoint(tmp_path)
        info = detect_checkpoint(ckpt_dir)
        assert info.has_optimizer_state is True
        assert any("pickle" in w.lower() for w in info.warnings)

    def test_non_checkpoint_directory(self, tmp_path: Path):
        """Non-checkpoint directory is not detected."""
        empty_dir = tmp_path / "not_a_checkpoint"
        empty_dir.mkdir()
        (empty_dir / "random.txt").write_text("hello", encoding="utf-8")

        info = detect_checkpoint(empty_dir)
        assert info.is_checkpoint is False

    def test_detected_files(self, tmp_path: Path):
        """Known checkpoint files are detected and categorized."""
        ckpt_dir = _make_checkpoint(tmp_path)
        info = detect_checkpoint(ckpt_dir)

        assert "trainer_state.json" in info.detected_files
        assert "training_args.bin" in info.detected_files
        assert "optimizer.pt" in info.detected_files
        assert "config.json" in info.detected_files


class TestBuildCheckpointManifestMetadata:
    """Tests for building checkpoint manifest metadata."""

    def test_build_metadata(self):
        """Build metadata from a detected checkpoint."""
        info = CheckpointInfo(
            is_checkpoint=True,
            step=1000,
            has_optimizer_state=True,
            has_scheduler_state=True,
            has_rng_state=True,
            has_trainer_state=True,
        )
        metadata = build_checkpoint_manifest_metadata(info)

        assert metadata["artifact_type"] == "training_checkpoint"
        assert metadata["step"] == 1000
        assert metadata["has_optimizer_state"] is True
        assert metadata["has_trainer_state"] is True

    def test_build_metadata_no_step(self):
        """Build metadata with unknown step."""
        info = CheckpointInfo(is_checkpoint=True, step=None)
        metadata = build_checkpoint_manifest_metadata(info)

        assert "step" not in metadata


class TestPackCheckpointRoundtrip:
    """Tests for pack-checkpoint roundtrip."""

    def test_pack_checkpoint_roundtrip(self, tmp_path: Path):
        """Pack a checkpoint and unpack it, verify byte-for-byte match."""
        ckpt_dir = _make_checkpoint(tmp_path)
        archive = tmp_path / "checkpoint.kmc"
        output_dir = tmp_path / "restored"

        from kmc.workflows.checkpoint import build_checkpoint_manifest_metadata

        info = detect_checkpoint(ckpt_dir)

        pack(
            ckpt_dir,
            archive,
            tensor_aware=info.has_safetensors_model,
            artifact_type="training_checkpoint",
            artifact_metadata=build_checkpoint_manifest_metadata(info),
        )

        assert archive.exists()

        unpack(archive, output_dir)

        # Compare files
        for original_file in ckpt_dir.rglob("*"):
            if original_file.is_file():
                rel = original_file.relative_to(ckpt_dir)
                restored_file = output_dir / rel
                assert restored_file.exists(), f"Missing: {rel}"
                assert original_file.read_bytes() == restored_file.read_bytes()

    def test_pack_checkpoint_manifest_has_artifact_type(self, tmp_path: Path):
        """Packed checkpoint archive has artifact_type='training_checkpoint'."""
        ckpt_dir = _make_checkpoint(tmp_path)
        archive = tmp_path / "checkpoint.kmc"

        from kmc.workflows.checkpoint import build_checkpoint_manifest_metadata

        info = detect_checkpoint(ckpt_dir)

        pack(
            ckpt_dir,
            archive,
            artifact_type="training_checkpoint",
            artifact_metadata=build_checkpoint_manifest_metadata(info),
        )

        from kmc.archive import inspect as inspect_archive

        manifest = inspect_archive(archive)
        assert manifest.artifact_type == "training_checkpoint"
        assert manifest.artifact_metadata.get("step") == 1000
