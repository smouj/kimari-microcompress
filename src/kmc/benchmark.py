"""Benchmark utilities for measuring KMC performance.

Compares raw, zlib, zstd and full KMC pack/unpack pipeline.
Supports console table, JSON output, file export, and optional
ZipNN comparison.

Environment metadata (Python version, OS, CPU, RAM, KMC version)
is included in JSON output for reproducibility.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify
from .codecs import _HAS_ZSTD, RawCodec, ZlibCodec, ZstdCodec
from .inspector import ModelFormat, inspect_directory

# ---------------------------------------------------------------------------
# Optional ZipNN detection
# ---------------------------------------------------------------------------

try:
    import zipnn as _zipnn  # type: ignore[import-untyped]

    _HAS_ZIPNN = True
except ImportError:
    _HAS_ZIPNN = False


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_KMC_VERSION = "0.3.0-alpha"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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
class ZipNNBenchmark:
    """Benchmark result for ZipNN comparison."""

    available: bool
    compressed_bytes: int = 0
    ratio: float = 0.0
    compress_seconds: float = 0.0
    decompress_seconds: float = 0.0
    version: str = ""


@dataclass
class EnvironmentInfo:
    """Environment information for reproducibility."""

    python_version: str = ""
    os_name: str = ""
    os_version: str = ""
    cpu: str = ""
    ram_gb: float = 0.0
    kmc_version: str = _KMC_VERSION
    zipnn_version: str = ""
    zstd_available: bool = False


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    tool: str = "kmc-bench"
    kmc_version: str = _KMC_VERSION
    source: str = ""
    synthetic: bool = False
    original_size: int = 0
    num_files: int = 0
    num_blocks: int = 0
    block_size: int = 0
    detected_formats: list[str] = field(default_factory=list)
    kmc_pack_time: float = 0.0
    kmc_unpack_time: float = 0.0
    kmc_verify_time: float = 0.0
    kmc_compressed_size: int = 0
    kmc_ratio: float = 0.0
    kmc_pack_throughput: float = 0.0
    kmc_unpack_throughput: float = 0.0
    codec_benchmarks: list[CodecBenchmark] = field(default_factory=list)
    zipnn_benchmark: ZipNNBenchmark | None = None
    environment: EnvironmentInfo | None = None
    tensor_aware: bool = False


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


def _get_environment_info() -> EnvironmentInfo:
    """Collect environment information for reproducibility."""
    ram_gb = 0.0
    try:
        import subprocess

        result = subprocess.run(["free", "-g"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 2:
                    ram_gb = float(parts[1])
    except Exception:
        pass

    # Try psutil as fallback
    if ram_gb == 0.0:
        try:
            import psutil  # type: ignore[import-untyped]

            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            pass

    zipnn_ver = ""
    if _HAS_ZIPNN:
        try:
            zipnn_ver = getattr(_zipnn, "__version__", "unknown")
        except Exception:
            zipnn_ver = "installed"

    return EnvironmentInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        os_name=platform.system(),
        os_version=platform.release(),
        cpu=platform.processor() or "unknown",
        ram_gb=ram_gb,
        kmc_version=_KMC_VERSION,
        zipnn_version=zipnn_ver,
        zstd_available=_HAS_ZSTD,
    )


# ---------------------------------------------------------------------------
# Codec benchmarks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ZipNN benchmark
# ---------------------------------------------------------------------------


def _run_zipnn_benchmark(source: Path, level: int = 3) -> ZipNNBenchmark:
    """Run ZipNN compression benchmark if available.

    Only benchmarks on compatible files (safetensors, .bin).
    Returns ZipNNBenchmark with available=False if ZipNN is not installed.
    """
    if not _HAS_ZIPNN:
        return ZipNNBenchmark(
            available=False,
        )

    try:
        import zipnn  # type: ignore[import-untyped]

        # Collect compatible files
        compatible_files: list[Path] = []
        if source.is_file():
            if source.suffix.lower() in (".safetensors", ".bin"):
                compatible_files.append(source)
        else:
            for f in source.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".safetensors", ".bin"):
                    compatible_files.append(f)

        if not compatible_files:
            return ZipNNBenchmark(available=True, ratio=0.0, version=zipnn.__version__)

        # Measure compression on compatible files
        total_original = 0
        total_compressed = 0
        total_compress_time = 0.0
        total_decompress_time = 0.0

        for f in compatible_files:
            data = f.read_bytes()
            total_original += len(data)

            # Compress
            t0 = time.perf_counter()
            compressed = zipnn.compress_data(data)  # type: ignore[attr-defined]
            total_compress_time += time.perf_counter() - t0
            total_compressed += len(compressed) if compressed else len(data)

            # Decompress
            if compressed and len(compressed) < len(data):
                t0 = time.perf_counter()
                try:
                    zipnn.decompress_data(compressed)  # type: ignore[attr-defined]
                    total_decompress_time += time.perf_counter() - t0
                except Exception:
                    total_decompress_time = 0.0

        ratio = total_compressed / total_original if total_original > 0 else 1.0

        return ZipNNBenchmark(
            available=True,
            compressed_bytes=total_compressed,
            ratio=ratio,
            compress_seconds=total_compress_time,
            decompress_seconds=total_decompress_time,
            version=zipnn.__version__,
        )
    except Exception as e:
        return ZipNNBenchmark(
            available=True,
            version=f"error: {e}",
        )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    source: Path,
    output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    synthetic: bool = False,
    tensor_aware: bool = False,
    compare_zipnn: bool = False,
) -> BenchmarkResult:
    """Run a complete benchmark: codec comparison + KMC pipeline + optional ZipNN.

    Args:
        source: Source directory or file to benchmark.
        output: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        synthetic: Whether the data is synthetic (mark in report).
        tensor_aware: Use tensor-aware compression mode.
        compare_zipnn: Compare with ZipNN if available.

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
    pack(source, output, block_size=block_size, level=level, tensor_aware=tensor_aware)
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

    # ZipNN benchmark (optional)
    zipnn_bench = None
    if compare_zipnn:
        zipnn_bench = _run_zipnn_benchmark(source, level=level)

    # Environment info
    env_info = _get_environment_info()

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
        zipnn_benchmark=zipnn_bench,
        environment=env_info,
        tensor_aware=tensor_aware,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_benchmark_table(result: BenchmarkResult) -> str:
    """Format a BenchmarkResult as a human-readable console table."""
    lines = [
        "=== KMC Benchmark Report ===",
        "",
        f"Source: {result.source}",
        f"KMC version: {result.kmc_version}",
        f"Data type: {'SYNTHETIC' if result.synthetic else 'REAL'}",
        f"Tensor-aware: {'yes' if result.tensor_aware else 'no'}",
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

    # ZipNN comparison
    if result.zipnn_benchmark is not None:
        zb = result.zipnn_benchmark
        lines.extend(["", "--- ZipNN Comparison ---"])
        if not zb.available:
            lines.append("  ZipNN: not available")
            lines.append("  Suggestion: pip install zipnn")
        else:
            lines.append(f"  ZipNN version: {zb.version or 'unknown'}")
            if zb.ratio > 0:
                lines.append(f"  ZipNN compressed: {zb.compressed_bytes:,} bytes")
                lines.append(f"  ZipNN ratio: {zb.ratio:.2%}")
                lines.append(f"  ZipNN compress time: {zb.compress_seconds:.3f}s")
                lines.append(f"  ZipNN decompress time: {zb.decompress_seconds:.3f}s")

                # Comparison note
                if result.kmc_ratio < zb.ratio:
                    lines.append(
                        f"  Note: KMC ratio ({result.kmc_ratio:.2%}) is better than "
                        f"ZipNN ({zb.ratio:.2%})"
                    )
                elif result.kmc_ratio > zb.ratio:
                    lines.append(
                        f"  Note: ZipNN ratio ({zb.ratio:.2%}) is better than "
                        f"KMC ({result.kmc_ratio:.2%})"
                    )
                else:
                    lines.append("  Note: KMC and ZipNN ratios are comparable")
            else:
                lines.append("  ZipNN: no compatible files found for benchmark")

        lines.append("  Disclaimer: This is a measurement, not a claim of superiority.")

    # Environment info
    if result.environment:
        env = result.environment
        lines.extend(
            [
                "",
                "--- Environment ---",
                f"  Python: {env.python_version}",
                f"  OS: {env.os_name} {env.os_version}",
                f"  CPU: {env.cpu}",
                f"  RAM: {env.ram_gb:.1f} GB" if env.ram_gb > 0 else "  RAM: unknown",
                f"  KMC: {env.kmc_version}",
                f"  zstd: {'available' if env.zstd_available else 'not available'}",
                f"  ZipNN: {env.zipnn_version or 'not available'}",
            ]
        )

    return "\n".join(lines)


def benchmark_to_json(result: BenchmarkResult) -> str:
    """Serialize a BenchmarkResult to JSON."""
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)
