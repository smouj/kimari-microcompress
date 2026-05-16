"""Benchmark utilities for measuring KMC performance.

Compares raw, zlib, zstd and full KMC pack/unpack pipeline.
Supports console table, JSON output and file export.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify
from .codecs import RawCodec, ZlibCodec, ZstdCodec, _HAS_ZSTD
from .inspector import ModelFormat, inspect_directory


@dataclass
class CodecBenchmark:
    """Benchmark result for a single codec."""

    codec: str
    original_size: int
    compressed_size: int
    ratio: float
    compress_time: float
    decompress_time: float
    compress_throughput: float  # bytes/s
    decompress_throughput: float  # bytes/s


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    source: str
    synthetic: bool
    original_size: int
    num_files: int
    num_blocks: int
    block_size: int
    detected_formats: list[str]
    kmc_pack_time: float
    kmc_unpack_time: float
    kmc_verify_time: float
    kmc_compressed_size: int
    kmc_ratio: float
    kmc_pack_throughput: float
    kmc_unpack_throughput: float
    codec_benchmarks: list[CodecBenchmark] = field(default_factory=list)


def _measure_codec(
    data: bytes,
    codec_name: str,
    level: int = 3,
) -> CodecBenchmark | None:
    """Measure compression and decompression for a single codec on data."""
    if codec_name == "zstd" and not _HAS_ZSTD:
        return None
    if codec_name == "zstd":
        codec = ZstdCodec()
    elif codec_name == "zlib":
        codec = ZlibCodec()
    elif codec_name == "raw":
        codec = RawCodec()
    else:
        return None

    # Compress
    t0 = time.perf_counter()
    result = codec.compress(data, level=level)
    compress_time = time.perf_counter() - t0

    # Only decompress if compression was effective
    if result.compressed_size < result.original_size or codec_name == "raw":
        t0 = time.perf_counter()
        codec.decompress(result.data, result.original_size)
        decompress_time = time.perf_counter() - t0
    else:
        decompress_time = 0.0

    ratio = result.compressed_size / result.original_size if result.original_size > 0 else 1.0
    compress_throughput = result.original_size / compress_time if compress_time > 0 else 0
    decompress_throughput = result.original_size / decompress_time if decompress_time > 0 else 0

    return CodecBenchmark(
        codec=codec_name,
        original_size=result.original_size,
        compressed_size=result.compressed_size,
        ratio=ratio,
        compress_time=compress_time,
        decompress_time=decompress_time,
        compress_throughput=compress_throughput,
        decompress_throughput=decompress_throughput,
    )


def run_benchmark(
    source: Path,
    output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    synthetic: bool = False,
) -> BenchmarkResult:
    """Run a complete benchmark: codec comparison + KMC pipeline.

    Args:
        source: Source directory or file to benchmark.
        output: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        synthetic: Whether the data is synthetic (mark in report).

    Returns:
        BenchmarkResult with all measurements.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()

    # Compute original size and file count
    if source.is_file():
        orig_size = source.stat().st_size
        num_files = 1
    else:
        files = [f for f in source.rglob("*") if f.is_file()]
        orig_size = sum(f.stat().st_size for f in files)
        num_files = len(files)

    # Detect model formats
    if source.is_dir():
        inspections = inspect_directory(source)
        detected_formats = sorted(
            set(i.format.value for i in inspections if i.format != ModelFormat.UNKNOWN)
        )
    else:
        from .inspector import inspect_file

        insp = inspect_file(source)
        detected_formats = [insp.format.value] if insp.format != ModelFormat.UNKNOWN else []

    # Per-codec benchmarks on first 1 MB of data
    codec_results: list[CodecBenchmark] = []
    sample_data = b""
    if source.is_file():
        with open(source, "rb") as f:
            sample_data = f.read(1024 * 1024)  # 1 MB sample
    else:
        for f in sorted(source.rglob("*")):
            if f.is_file() and len(sample_data) < 1024 * 1024:
                with open(f, "rb") as fh:
                    sample_data += fh.read(1024 * 1024 - len(sample_data))

    for codec_name in ["raw", "zlib", "zstd"]:
        result = _measure_codec(sample_data, codec_name, level=level)
        if result is not None:
            codec_results.append(result)

    # KMC pack
    t0 = time.perf_counter()
    pack(source, output, block_size=block_size, level=level)
    pack_time = time.perf_counter() - t0
    comp_size = output.stat().st_size

    # KMC verify
    t0 = time.perf_counter()
    verify(output)
    verify_time = time.perf_counter() - t0

    # KMC unpack
    unpack_time = 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        restore_dir = Path(tmpdir) / "restored"
        t0 = time.perf_counter()
        unpack(output, restore_dir)
        unpack_time = time.perf_counter() - t0

    ratio = comp_size / orig_size if orig_size > 0 else 0
    pack_throughput = orig_size / pack_time if pack_time > 0 else 0
    unpack_throughput = orig_size / unpack_time if unpack_time > 0 else 0

    # Estimate block count from the archive manifest
    from .archive import read_manifest_from_archive

    manifest, _ = read_manifest_from_archive(output)
    num_blocks = sum(len(f.blocks) for f in manifest.files)

    return BenchmarkResult(
        source=str(source),
        synthetic=synthetic,
        original_size=orig_size,
        num_files=num_files,
        num_blocks=num_blocks,
        block_size=block_size,
        detected_formats=detected_formats,
        kmc_pack_time=pack_time,
        kmc_unpack_time=unpack_time,
        kmc_verify_time=verify_time,
        kmc_compressed_size=comp_size,
        kmc_ratio=ratio,
        kmc_pack_throughput=pack_throughput,
        kmc_unpack_throughput=unpack_throughput,
        codec_benchmarks=codec_results,
    )


def format_benchmark_table(result: BenchmarkResult) -> str:
    """Format a BenchmarkResult as a human-readable console table."""
    lines = [
        "=== KMC Benchmark Report ===",
        "",
        f"Source: {result.source}",
        f"Data type: {'SYNTHETIC' if result.synthetic else 'REAL'}",
        f"Original size: {result.original_size:,} bytes",
        f"Files: {result.num_files}",
        f"Blocks: {result.num_blocks} (block_size={result.block_size:,})",
        f"Detected formats: {', '.join(result.detected_formats) or 'none'}",
        "",
        "--- Codec Comparison (1 MB sample) ---",
    ]

    # Table header
    lines.append(
        f"  {'Codec':<8} {'Compressed':>12} {'Ratio':>8} "
        f"{'Comp(s)':>10} {'Decomp(s)':>10} "
        f"{'CompMB/s':>10} {'DecompMB/s':>10}"
    )
    lines.append(f"  {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for cb in result.codec_benchmarks:
        lines.append(
            f"  {cb.codec:<8} {cb.compressed_size:>12,} {cb.ratio:>7.2%} "
            f"{cb.compress_time:>10.4f} {cb.decompress_time:>10.4f} "
            f"{cb.compress_throughput / 1024 / 1024:>10.2f} "
            f"{cb.decompress_throughput / 1024 / 1024:>10.2f}"
        )

    lines.extend(
        [
            "",
            "--- KMC Pipeline ---",
            f"  Compressed size: {result.kmc_compressed_size:,} bytes",
            f"  Ratio: {result.kmc_ratio:.2%}",
            f"  Pack time:   {result.kmc_pack_time:.3f}s "
            f"({result.kmc_pack_throughput / 1024 / 1024:.2f} MB/s)",
            f"  Unpack time: {result.kmc_unpack_time:.3f}s "
            f"({result.kmc_unpack_throughput / 1024 / 1024:.2f} MB/s)",
            f"  Verify time: {result.kmc_verify_time:.3f}s",
        ]
    )

    return "\n".join(lines)


def benchmark_to_json(result: BenchmarkResult) -> str:
    """Serialize a BenchmarkResult to JSON."""
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)
