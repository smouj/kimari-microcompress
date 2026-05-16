"""Tests for v0.5 CLI commands: pack-lora, pack-checkpoint, inspect flags."""

from pathlib import Path

import pytest

from kmc.cli import build_parser


class TestCLIPackLora:
    """Tests for pack-lora CLI command."""

    def test_pack_lora_help(self):
        """pack-lora --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["pack-lora", "--help"])
        assert exc_info.value.code == 0

    def test_pack_lora_requires_source_and_output(self):
        """pack-lora requires source and output arguments."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["pack-lora"])


class TestCLIPackCheckpoint:
    """Tests for pack-checkpoint CLI command."""

    def test_pack_checkpoint_help(self):
        """pack-checkpoint --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["pack-checkpoint", "--help"])
        assert exc_info.value.code == 0

    def test_pack_checkpoint_requires_source_and_output(self):
        """pack-checkpoint requires source and output arguments."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["pack-checkpoint"])


class TestCLIInspectFlags:
    """Tests for inspect CLI flags."""

    def test_inspect_lora_flag(self):
        """inspect --lora flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["inspect", "path", "--lora"])
        assert args.lora is True

    def test_inspect_checkpoint_flag(self):
        """inspect --checkpoint flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["inspect", "path", "--checkpoint"])
        assert args.checkpoint is True

    def test_inspect_gguf_flag(self):
        """inspect --gguf flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["inspect", "path", "--gguf"])
        assert args.gguf is True

    def test_inspect_gguf_tensors_json(self):
        """inspect --gguf --tensors --json flags are all parsed."""
        parser = build_parser()
        args = parser.parse_args(["inspect", "path", "--gguf", "--tensors", "--json"])
        assert args.gguf is True
        assert args.tensors is True
        assert args.json is True

    def test_inspect_compression_flag(self):
        """inspect --compression flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["inspect", "path", "--compression"])
        assert args.compression is True


class TestCLIPackGgufAware:
    """Tests for pack --gguf-aware flag."""

    def test_gguf_aware_flag(self):
        """pack --gguf-aware flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["pack", "src", "out.kmc", "--gguf-aware"])
        assert args.gguf_aware is True

    def test_gguf_aware_default_false(self):
        """pack --gguf-aware defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["pack", "src", "out.kmc"])
        assert args.gguf_aware is False


class TestCLIHelpCommands:
    """Tests that all help commands work."""

    def test_main_help(self):
        """kmc --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_inspect_help(self):
        """kmc inspect --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["inspect", "--help"])
        assert exc_info.value.code == 0

    def test_bench_help(self):
        """kmc bench --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["bench", "--help"])
        assert exc_info.value.code == 0

    def test_pack_help(self):
        """kmc pack --help works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["pack", "--help"])
        assert exc_info.value.code == 0


class TestGGUFAwarePackRoundtrip:
    """Tests for GGUF-aware compression mode."""

    def test_gguf_aware_pack_roundtrip(self, tmp_path: Path):
        """Pack a GGUF file with --gguf-aware and verify roundtrip."""
        import struct

        from kmc.archive import pack, unpack
        from kmc.formats.gguf import GGUF_MAGIC_LE

        # Create a synthetic GGUF file
        gguf_path = tmp_path / "model.gguf"
        with open(gguf_path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC_LE))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 2))  # tensor_count
            f.write(struct.pack("<Q", 0))  # kv_count
            # Write 2 tensor info entries
            for name in [b"token_embd.weight", b"output.weight"]:
                f.write(struct.pack("<Q", len(name)))
                f.write(name)
                f.write(struct.pack("<I", 1))  # 1 dimension
                f.write(struct.pack("<Q", 4))  # shape
                f.write(struct.pack("<I", 14))  # Q6_K
                f.write(struct.pack("<Q", 0))  # offset
            f.write(b"\x00" * 512)

        archive = tmp_path / "model.gguf.kmc"
        output_dir = tmp_path / "restored"

        pack(gguf_path, archive, gguf_aware=True)
        assert archive.exists()

        unpack(archive, output_dir)

        # Verify roundtrip
        original = gguf_path.read_bytes()
        restored = (output_dir / "model.gguf").read_bytes()
        assert original == restored

    def test_gguf_aware_manifest_has_artifact_type(self, tmp_path: Path):
        """GGUF-aware packed archive has artifact_type='gguf_model'."""
        import struct

        from kmc.archive import pack, inspect as inspect_archive
        from kmc.formats.gguf import GGUF_MAGIC_LE

        gguf_path = tmp_path / "model.gguf"
        with open(gguf_path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC_LE))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<Q", 0))
            name = b"test.weight"
            f.write(struct.pack("<Q", len(name)))
            f.write(name)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", 4))
            f.write(struct.pack("<I", 2))  # Q4_0 (quantized)
            f.write(struct.pack("<Q", 0))
            f.write(b"\x00" * 256)

        archive = tmp_path / "model.gguf.kmc"
        pack(gguf_path, archive, gguf_aware=True)

        manifest = inspect_archive(archive)
        assert manifest.artifact_type == "gguf_model"
        assert "gguf" in manifest.format_metadata
