"""Tests for tensor-aware packing, improved inspect, and benchmark with ZipNN."""

import io
import json
import struct
from contextlib import redirect_stdout
from pathlib import Path

from kmc.archive import pack, read_manifest_from_archive, unpack, verify
from kmc.manifest import BlockEntry, FileEntry, KMCManifest, TensorEntry


def _make_safetensors_file(
    path: Path,
    tensors: dict | None = None,
    metadata: dict | None = None,
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
        f.write(b"\x00" * data_size)

    return path


# ---------------------------------------------------------------------------
# Tensor-aware packing tests
# ---------------------------------------------------------------------------


class TestTensorAwarePacking:
    """Tests for the --tensor-aware packing mode."""

    def test_tensor_aware_pack_produces_valid_archive(self, tmp_path: Path):
        """Tensor-aware pack produces a valid archive that passes verification."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        _make_safetensors_file(source_dir / "model.safetensors")
        (source_dir / "config.json").write_text('{"model_type": "test"}')

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=True)

        errors = verify(archive)
        assert errors == []

    def test_tensor_aware_roundtrip(self, tmp_path: Path):
        """Tensor-aware pack/unpack roundtrip preserves all bytes."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        _make_safetensors_file(source_dir / "model.safetensors")
        (source_dir / "config.json").write_text('{"model_type": "test"}')

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=True)

        restore_dir = tmp_path / "restored"
        unpack(archive, restore_dir)

        for f in source_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(source_dir)
                restored = restore_dir / rel
                assert restored.exists(), f"Missing restored file: {rel}"
                assert restored.read_bytes() == f.read_bytes(), f"Mismatch: {rel}"

    def test_tensor_aware_manifest_has_tensor_entries(self, tmp_path: Path):
        """Tensor-aware pack adds tensor entries to the manifest."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        _make_safetensors_file(source_dir / "model.safetensors")

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=True)

        manifest, _ = read_manifest_from_archive(archive)

        # Find the safetensors file entry
        st_entry = None
        for f in manifest.files:
            if f.path.endswith(".safetensors"):
                st_entry = f
                break

        assert st_entry is not None
        assert st_entry.tensor_count > 0
        assert len(st_entry.tensor_entries) > 0
        assert st_entry.dtype_summary  # Has dtype info

    def test_tensor_aware_manifest_version(self, tmp_path: Path):
        """Tensor-aware pack uses manifest version 2."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        _make_safetensors_file(source_dir / "model.safetensors")

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=True)

        manifest, _ = read_manifest_from_archive(archive)
        assert manifest.version >= 2  # v2 or later for tensor-aware

    def test_non_safetensors_tensor_aware(self, tmp_path: Path):
        """Tensor-aware mode works with non-safetensors files (falls back)."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02\x03" * 5000)

        archive = tmp_path / "test.kmc"
        pack(source, archive, tensor_aware=True)

        errors = verify(archive)
        assert errors == []

        # Non-safetensors file should have no tensor entries
        manifest, _ = read_manifest_from_archive(archive)
        for f in manifest.files:
            if f.path.endswith(".bin"):
                assert f.tensor_count == 0
                assert f.tensor_entries == []

    def test_tensor_aware_no_tensor_entries_for_non_safetensors(self, tmp_path: Path):
        """Non-safetensors files have zero tensor_count in tensor-aware mode."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.json").write_text('{"model_type": "test"}')

        archive = tmp_path / "test.kmc"
        pack(source, archive, tensor_aware=True)

        manifest, _ = read_manifest_from_archive(archive)
        for f in manifest.files:
            assert f.tensor_count == 0
            assert f.tensor_entries == []


# ---------------------------------------------------------------------------
# v0.2 backward compatibility tests
# ---------------------------------------------------------------------------


class TestV02Compatibility:
    """Tests for backward compatibility with v0.2 .kmc archives."""

    def test_v1_manifest_read_by_v2_code(self, tmp_path: Path):
        """A v1 manifest (no tensor fields) is read correctly by v2 code."""
        source = tmp_path / "source.txt"
        source.write_bytes(b"Compatibility test " * 1000)

        archive = tmp_path / "test.kmc"
        pack(source, archive, tensor_aware=False)

        manifest, _ = read_manifest_from_archive(archive)
        # Should read without errors
        assert len(manifest.files) == 1
        # Tensor fields should default to empty/zero
        assert manifest.files[0].tensor_count == 0
        assert manifest.files[0].tensor_entries == []

    def test_v1_archive_roundtrip_with_v2_code(self, tmp_path: Path):
        """A v1 archive packs and unpacks correctly with v2 code."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "model.bin").write_bytes(b"\xaa\xbb\xcc" * 5000)

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=False)

        restore_dir = tmp_path / "restored"
        unpack(archive, restore_dir)

        assert (restore_dir / "model.bin").exists()
        assert (restore_dir / "model.bin").read_bytes() == b"\xaa\xbb\xcc" * 5000


# ---------------------------------------------------------------------------
# kmc inspect --json tests
# ---------------------------------------------------------------------------


class TestInspectJSON:
    """Tests for the kmc inspect --json output."""

    def test_inspect_model_json_output(self, tmp_path: Path):
        """kmc inspect --json produces valid JSON for a model directory."""
        from kmc.cli import _inspect_model

        source_dir = tmp_path / "model"
        source_dir.mkdir()
        _make_safetensors_file(source_dir / "model.safetensors")
        (source_dir / "config.json").write_text('{"model_type": "test"}')

        buf = io.StringIO()
        with redirect_stdout(buf):
            _inspect_model(source_dir, json_output=True, show_tensors=False)

        output = buf.getvalue()
        parsed = json.loads(output)
        assert "path" in parsed
        assert "files" in parsed
        assert (
            parsed.get("type") in ("model_directory", "model_file")
            or parsed.get("artifact_type") is not None
        )

    def test_inspect_archive_json_output(self, tmp_path: Path):
        """kmc inspect --json produces valid JSON for a .kmc archive."""
        from kmc.cli import _inspect_archive

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"Test " * 1000)

        archive = tmp_path / "test.kmc"
        pack(source, archive)

        buf = io.StringIO()
        with redirect_stdout(buf):
            _inspect_archive(archive, json_output=True, show_tensors=False)

        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["type"] == "kmc_archive"
        assert "files" in parsed
        assert "version" in parsed

    def test_inspect_json_with_tensors(self, tmp_path: Path):
        """kmc inspect --json --tensors includes tensor details."""
        from kmc.cli import _inspect_archive

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _make_safetensors_file(source_dir / "model.safetensors")

        archive = tmp_path / "test.kmc"
        pack(source_dir, archive, tensor_aware=True)

        buf = io.StringIO()
        with redirect_stdout(buf):
            _inspect_archive(archive, json_output=True, show_tensors=True)

        output = buf.getvalue()
        parsed = json.loads(output)
        assert "files" in parsed
        # The safetensors file should have tensor entries
        st_file = next(f for f in parsed["files"] if f["path"].endswith(".safetensors"))
        assert st_file.get("tensor_count", 0) > 0


# ---------------------------------------------------------------------------
# Benchmark with ZipNN absent tests
# ---------------------------------------------------------------------------


class TestBenchmarkZipNNAbsent:
    """Tests for benchmark behavior when ZipNN is not installed."""

    def test_benchmark_without_zipnn(self, tmp_path: Path):
        """Benchmark runs successfully without ZipNN."""
        from kmc.benchmark import run_benchmark

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 10000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True, compare_zipnn=False)

        assert result.kmc_compressed_size > 0
        assert result.zipnn_benchmark is None  # Not requested

    def test_benchmark_with_zipnn_compare_but_absent(self, tmp_path: Path):
        """Benchmark with compare_zipnn=True but ZipNN not installed."""
        from kmc.benchmark import _HAS_ZIPNN, run_benchmark

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 10000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True, compare_zipnn=True)

        # If ZipNN is not installed, should still work
        assert result.zipnn_benchmark is not None
        if not _HAS_ZIPNN:
            assert result.zipnn_benchmark.available is False
        else:
            # If it is installed, should have results
            assert result.zipnn_benchmark.available is True

    def test_benchmark_zipnn_json_output(self, tmp_path: Path):
        """Benchmark JSON output includes ZipNN section when requested."""
        from kmc.benchmark import benchmark_to_json, run_benchmark

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 10000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True, compare_zipnn=True)

        json_str = benchmark_to_json(result)
        parsed = json.loads(json_str)
        assert "zipnn_benchmark" in parsed
        assert "environment" in parsed

    def test_benchmark_environment_info(self, tmp_path: Path):
        """Benchmark includes environment information."""
        from kmc.benchmark import run_benchmark

        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 10000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True)

        assert result.environment is not None
        assert result.environment.python_version != ""
        assert result.environment.kmc_version == "0.5.0-alpha"

    def test_benchmark_tensor_aware_flag(self, tmp_path: Path):
        """Benchmark with tensor_aware flag."""
        from kmc.benchmark import run_benchmark

        source = tmp_path / "source"
        source.mkdir()
        _make_safetensors_file(source / "model.safetensors")

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True, tensor_aware=True)

        assert result.tensor_aware is True
        assert result.kmc_compressed_size > 0


# ---------------------------------------------------------------------------
# Manifest v2 compatibility tests
# ---------------------------------------------------------------------------


class TestManifestV2Compat:
    """Tests for manifest v2 (tensor-aware) format."""

    def test_manifest_v2_roundtrip(self):
        """Manifest v2 with tensor entries serializes and deserializes correctly."""
        tensor_entry = TensorEntry(
            name="weight.weight",
            dtype="BF16",
            shape=[4096, 4096],
            byte_offset=0,
            byte_size=33554432,
        )
        block = BlockEntry(
            index=0,
            offset=100,
            compressed_size=200,
            original_size=300,
            codec="zstd",
            hash="abc123",
        )
        file_entry = FileEntry(
            path="model.safetensors",
            original_size=33554432,
            hash="def456",
            block_size=262144,
            blocks=[block],
            tensor_count=1,
            dtype_summary=["BF16"],
            tensor_entries=[tensor_entry],
        )
        manifest = KMCManifest(
            version=2,
            files=[file_entry],
            total_original_size=33554432,
            total_compressed_size=20000000,
        )

        json_str = manifest.to_json()
        restored = KMCManifest.from_json(json_str)

        assert restored.version == 2
        assert len(restored.files) == 1
        assert restored.files[0].tensor_count == 1
        assert len(restored.files[0].tensor_entries) == 1
        assert restored.files[0].tensor_entries[0].name == "weight.weight"
        assert restored.files[0].tensor_entries[0].dtype == "BF16"
        assert restored.files[0].dtype_summary == ["BF16"]

    def test_manifest_v1_read_as_v2(self):
        """A v1 manifest (without tensor fields) is read correctly."""
        v1_json = json.dumps(
            {
                "version": 1,
                "tool": "kimari-microcompress",
                "tool_version": "0.1.0",
                "created_at": "2024-01-01T00:00:00",
                "total_original_size": 1000,
                "total_compressed_size": 800,
                "files": [
                    {
                        "path": "model.bin",
                        "original_size": 1000,
                        "hash": "abc123",
                        "block_size": 262144,
                        "blocks": [
                            {
                                "index": 0,
                                "offset": 100,
                                "compressed_size": 800,
                                "original_size": 1000,
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
        # Tensor fields should default to empty
        assert manifest.files[0].tensor_count == 0
        assert manifest.files[0].tensor_entries == []
        assert manifest.files[0].dtype_summary == []
