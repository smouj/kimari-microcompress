#!/usr/bin/env python3
"""Reproducible benchmark script for GGUF model files.

Usage:
    python scripts/bench_real_gguf.py ./models/tiny.gguf --output reports/tiny-gguf.md

This script runs KMC benchmarks on a local GGUF file with --gguf-aware mode.
It does NOT download models automatically.
It generates JSON, Markdown table, and environment metadata.
It does NOT invent results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kmc.archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify_quick
from kmc.benchmark import _get_environment_info


def run_gguf_benchmark(gguf_path: Path, output_path: Path | None = None) -> dict:
    """Run benchmark on a GGUF file.

    Args:
        gguf_path: Path to the GGUF file.
        output_path: Optional path to write Markdown report.

    Returns:
        Dict with benchmark results.
    """
    if not gguf_path.exists():
        print(f"Error: GGUF file not found: {gguf_path}", file=sys.stderr)
        sys.exit(1)

    if not gguf_path.is_file():
        print(
            f"Error: expected a file, got directory: {gguf_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    original_size = gguf_path.stat().st_size
    print(f"Benchmarking GGUF: {gguf_path}")
    print(f"Size: {original_size:,} bytes")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "gguf_benchmark.kmc"

        # Pack with gguf_aware
        t0 = time.perf_counter()
        pack(
            gguf_path,
            archive_path,
            block_size=DEFAULT_BLOCK_SIZE,
            gguf_aware=True,
        )
        pack_time = time.perf_counter() - t0

        compressed_size = archive_path.stat().st_size
        ratio = compressed_size / original_size if original_size > 0 else 0

        # Pack without gguf_aware for comparison
        archive_naive = Path(tmpdir) / "gguf_naive.kmc"
        t0 = time.perf_counter()
        pack(
            gguf_path,
            archive_naive,
            block_size=DEFAULT_BLOCK_SIZE,
            gguf_aware=False,
        )
        naive_pack_time = time.perf_counter() - t0
        naive_compressed_size = archive_naive.stat().st_size
        naive_ratio = naive_compressed_size / original_size if original_size > 0 else 0

        # Verify
        t0 = time.perf_counter()
        verify_quick(archive_path)
        verify_time = time.perf_counter() - t0

        # Unpack
        restore_dir = Path(tmpdir) / "restored"
        t0 = time.perf_counter()
        unpack(archive_path, restore_dir)
        unpack_time = time.perf_counter() - t0

    env_info = _get_environment_info()

    results = {
        "gguf_path": str(gguf_path),
        "original_size": original_size,
        "gguf_aware": {
            "compressed_size": compressed_size,
            "ratio": ratio,
            "pack_time_s": round(pack_time, 4),
        },
        "naive": {
            "compressed_size": naive_compressed_size,
            "ratio": naive_ratio,
            "pack_time_s": round(naive_pack_time, 4),
        },
        "verify_time_s": round(verify_time, 4),
        "unpack_time_s": round(unpack_time, 4),
        "environment": {
            "python_version": env_info.python_version,
            "os": f"{env_info.os_name} {env_info.os_version}",
            "cpu": env_info.cpu,
            "ram_gb": env_info.ram_gb,
            "kmc_version": env_info.kmc_version,
            "zstd_available": env_info.zstd_available,
        },
        "reproducibility_warning": (
            "Results are machine-specific. Run on your own hardware for accurate numbers."
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown_report(results, output_path)
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Report saved to {output_path}")
        print(f"JSON saved to {json_path}")

    return results


def _write_markdown_report(results: dict, path: Path) -> None:
    """Write a Markdown benchmark report for GGUF."""
    env = results["environment"]
    ga = results["gguf_aware"]
    na = results["naive"]
    lines = [
        "# KMC GGUF Benchmark Report",
        "",
        f"**File**: `{results['gguf_path']}`",
        f"**KMC Version**: {env['kmc_version']}",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## GGUF-aware vs Naive",
        "",
        "| Metric | GGUF-aware | Naive |",
        "|--------|-----------|-------|",
        f"| Compressed size | {ga['compressed_size']:,} bytes | {na['compressed_size']:,} bytes |",
        f"| Ratio | {ga['ratio']:.2%} | {na['ratio']:.2%} |",
        f"| Pack time | {ga['pack_time_s']:.3f}s | {na['pack_time_s']:.3f}s |",
        "",
        "## General",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Original size | {results['original_size']:,} bytes |",
        f"| Verify time | {results['verify_time_s']:.3f}s |",
        f"| Unpack time | {results['unpack_time_s']:.3f}s |",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Python | {env['python_version']} |",
        f"| OS | {env['os']} |",
        f"| CPU | {env['cpu']} |",
        f"| RAM | {env['ram_gb']:.1f} GB |",
        f"| zstd | {'available' if env['zstd_available'] else 'not available'} |",
        "",
        "## Reproducibility Notice",
        "",
        "These results are specific to the hardware and software environment listed above.",
        "Do not compare across different machines without normalization.",
        "This benchmark was run locally and has not been independently verified.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KMC on a GGUF file")
    parser.add_argument("gguf_path", type=Path, help="Path to local GGUF file")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown report path")
    args = parser.parse_args()

    run_gguf_benchmark(args.gguf_path, args.output)


if __name__ == "__main__":
    main()
