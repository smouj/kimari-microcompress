"""Kimari CLI integration: maps kimari commands to KMC operations.

This module provides a clean adapter layer that maps Kimari CLI commands
to the underlying KMC operations:

    kimari compress       -> kmc pack [--tensor-aware] [--codec]
    kimari decompress     -> kmc unpack
    kimari verify-compress -> kmc verify
    kimari bench-compress -> kmc bench [--compare-zipnn] [--compare-codecs]

Usage from Kimari CLI:

    from kmc.integrations.kimari import (
        kimari_compress,
        kimari_decompress,
        kimari_verify_compress,
        kimari_bench_compress,
    )

This module does NOT modify Kimari itself. It provides the integration
surface that Kimari can call when the KMC package is available.
"""

from __future__ import annotations

from pathlib import Path

from ..archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify_full
from ..benchmark import BenchmarkResult, run_benchmark


def kimari_compress(
    source: str | Path,
    output: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    tensor_aware: bool = True,
    codec: str = "auto",
) -> dict:
    """Compress a model using KMC (kimari compress).

    Args:
        source: Source directory or file.
        output: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        tensor_aware: Use tensor-aware mode (default True for Kimari).
        codec: Compression codec ('auto', 'byteplane', 'floatplane',
            'zstd', 'zlib', 'raw').

    Returns:
        Dict with status, original and compressed sizes, and ratio.
    """
    source = Path(source)
    output = Path(output)

    pack(
        source,
        output,
        block_size=block_size,
        level=level,
        tensor_aware=tensor_aware,
        codec=codec,
    )

    if source.is_file():
        orig_size = source.stat().st_size
    else:
        orig_size = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())

    comp_size = output.stat().st_size
    ratio = comp_size / orig_size if orig_size > 0 else 0

    return {
        "status": "ok",
        "source": str(source),
        "output": str(output),
        "original_size": orig_size,
        "compressed_size": comp_size,
        "ratio": ratio,
        "tensor_aware": tensor_aware,
        "codec": codec,
    }


def kimari_decompress(
    archive: str | Path,
    output_dir: str | Path,
) -> dict:
    """Decompress a .kmc archive (kimari decompress).

    Args:
        archive: Path to the .kmc archive.
        output_dir: Output directory.

    Returns:
        Dict with status and output path.
    """
    archive = Path(archive)
    output_dir = Path(output_dir)

    unpack(archive, output_dir)

    return {
        "status": "ok",
        "archive": str(archive),
        "output_dir": str(output_dir),
    }


def kimari_verify_compress(archive: str | Path) -> dict:
    """Verify a .kmc archive's integrity (kimari verify-compress).

    Args:
        archive: Path to the .kmc archive.

    Returns:
        Dict with status, integrity result, and any errors.
    """
    archive = Path(archive)
    report = verify_full(archive)

    return {
        "status": "ok" if report.integrity == "OK" else "failed",
        "archive": str(archive),
        "integrity": report.integrity,
        "errors": report.errors,
        "warnings": report.warnings,
        "total_files": report.total_files,
        "total_blocks": report.total_blocks,
    }


def kimari_bench_compress(
    source: str | Path,
    output: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    synthetic: bool = False,
    tensor_aware: bool = True,
    compare_zipnn: bool = False,
    compare_codecs: bool = False,
    codec: str = "auto",
) -> BenchmarkResult:
    """Benchmark compression (kimari bench-compress).

    Args:
        source: Source directory or file.
        output: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        synthetic: Whether the data is synthetic.
        tensor_aware: Use tensor-aware mode (default True for Kimari).
        compare_zipnn: Compare with ZipNN if available.
        compare_codecs: Compare all available codecs.
        codec: Compression codec for the main pipeline.

    Returns:
        BenchmarkResult with all measurements.
    """
    return run_benchmark(
        Path(source),
        Path(output),
        block_size=block_size,
        level=level,
        synthetic=synthetic,
        tensor_aware=tensor_aware,
        compare_zipnn=compare_zipnn,
        compare_codecs=compare_codecs,
        codec=codec,
    )


# Command mapping for documentation
KIMARI_COMMAND_MAP = {
    "kimari compress": (
        "kmc pack [--tensor-aware] [--codec auto|byteplane|floatplane|zstd|zlib|raw]"
    ),
    "kimari decompress": "kmc unpack",
    "kimari verify-compress": "kmc verify",
    "kimari bench-compress": "kmc bench [--compare-zipnn] [--compare-codecs]",
}
