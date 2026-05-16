"""Tests for benchmark and inspector improvements."""

import json
from pathlib import Path

from kmc.benchmark import (
    BenchmarkResult,
    benchmark_to_json,
    format_benchmark_table,
    run_benchmark,
)
from kmc.inspector import (
    ModelFormat,
    inspect_directory,
    inspect_file,
)


class TestBenchmark:
    """Tests for the benchmark system."""

    def test_run_benchmark_returns_result(self, tmp_path: Path):
        """run_benchmark returns a BenchmarkResult with expected fields."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"\x00\x01\x02" * 10000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output, synthetic=True)

        assert isinstance(result, BenchmarkResult)
        assert result.original_size > 0
        assert result.kmc_compressed_size > 0
        assert result.kmc_pack_time > 0
        assert result.kmc_unpack_time > 0
        assert result.num_files == 1
        assert result.synthetic is True
        assert len(result.codec_benchmarks) > 0

    def test_benchmark_json_output(self, tmp_path: Path):
        """benchmark_to_json produces valid JSON."""
        source = tmp_path / "source.txt"
        source.write_bytes(b"JSON test " * 5000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output)

        json_str = benchmark_to_json(result)
        parsed = json.loads(json_str)
        assert "source" in parsed
        assert "kmc_ratio" in parsed
        assert "codec_benchmarks" in parsed

    def test_benchmark_table_output(self, tmp_path: Path):
        """format_benchmark_table produces readable output."""
        source = tmp_path / "source.txt"
        source.write_bytes(b"Table test " * 5000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output)

        table = format_benchmark_table(result)
        assert "KMC Benchmark Report" in table
        assert "Codec" in table

    def test_benchmark_file_export(self, tmp_path: Path):
        """Benchmark results can be saved to a JSON file."""
        source = tmp_path / "source.txt"
        source.write_bytes(b"Export test " * 5000)

        output = tmp_path / "test.kmc"
        result = run_benchmark(source, output)

        out_file = tmp_path / "bench.json"
        out_file.write_text(benchmark_to_json(result))

        assert out_file.exists()
        parsed = json.loads(out_file.read_text())
        assert parsed["kmc_ratio"] > 0


class TestInspector:
    """Tests for the improved AI model inspector."""

    def test_detect_safetensors(self, tmp_path: Path):
        """Inspector detects safetensors files and reads tensor metadata."""
        import struct

        header_dict = {
            "__metadata__": {"model_type": "test"},
            "weight1.weight": {"dtype": "F32", "shape": [64, 64], "data_offsets": [0, 16384]},
            "weight2.bias": {"dtype": "F16", "shape": [64], "data_offsets": [16384, 17152]},
        }
        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = struct.pack("<Q", len(header_json))

        st_file = tmp_path / "model.safetensors"
        with open(st_file, "wb") as f:
            f.write(header_len)
            f.write(header_json)
            f.write(b"\x00" * 17152)

        result = inspect_file(st_file)
        assert result.format == ModelFormat.SAFETENSORS
        assert len(result.tensors) == 2
        assert result.tensors[0].name == "weight1.weight"
        assert result.tensors[0].dtype == "F32"
        assert result.extra.get("total_params") == 64 * 64 + 64

    def test_detect_config_json(self, tmp_path: Path):
        """Inspector detects config.json files."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"model_type": "test"}')

        result = inspect_file(config_file)
        assert result.format == ModelFormat.CONFIG

    def test_detect_tokenizer(self, tmp_path: Path):
        """Inspector detects tokenizer files."""
        tok_file = tmp_path / "tokenizer.json"
        tok_file.write_text('{"version": "1.0"}')

        result = inspect_file(tok_file)
        assert result.format == ModelFormat.TOKENIZER

    def test_detect_shard(self, tmp_path: Path):
        """Inspector detects model shard files."""
        shard = tmp_path / "model-00001-of-00002.safetensors"
        shard.write_bytes(b"\x00" * 100)

        result = inspect_file(shard)
        assert result.format == ModelFormat.SHARD

    def test_detect_pytorch_bin(self, tmp_path: Path):
        """Inspector detects PyTorch .bin files with pickle magic."""
        bin_file = tmp_path / "pytorch_model.bin"
        bin_file.write_bytes(b"\x80\x02" + b"\x00" * 100)

        result = inspect_file(bin_file)
        assert result.format == ModelFormat.PYTORCH_BIN

    def test_detect_lora_adapter(self, tmp_path: Path):
        """Inspector detects LoRA adapters by tensor names."""
        import struct

        header_dict = {
            "lora_A.weight": {"dtype": "F32", "shape": [8, 64], "data_offsets": [0, 2048]},
            "lora_B.weight": {"dtype": "F32", "shape": [64, 8], "data_offsets": [2048, 4096]},
        }
        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = struct.pack("<Q", len(header_json))

        lora_file = tmp_path / "adapter_model.safetensors"
        with open(lora_file, "wb") as f:
            f.write(header_len)
            f.write(header_json)
            f.write(b"\x00" * 4096)

        result = inspect_file(lora_file)
        assert result.format == ModelFormat.LORA_ADAPTER
        assert len(result.extra.get("lora_tensors", [])) >= 2

    def test_inspect_directory(self, tmp_path: Path):
        """Inspector scans a directory and returns results for all files."""
        (tmp_path / "config.json").write_text('{"model_type": "test"}')
        (tmp_path / "tokenizer.json").write_text('{"version": "1.0"}')
        (tmp_path / "README.md").write_text("# Model")

        results = inspect_directory(tmp_path)
        formats = {r.format for r in results}
        assert ModelFormat.CONFIG in formats
        assert ModelFormat.TOKENIZER in formats
        assert ModelFormat.UNKNOWN in formats  # README.md

    def test_unknown_format(self, tmp_path: Path):
        """Inspector returns UNKNOWN for unrecognized files."""
        unknown_file = tmp_path / "data.xyz"
        unknown_file.write_bytes(b"\x00" * 100)

        result = inspect_file(unknown_file)
        assert result.format == ModelFormat.UNKNOWN
