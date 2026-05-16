"""Benchmark utilities for measuring KMC performance."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify


def run_benchmark(
    source: Path,
    output: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    """Run a complete benchmark: pack, verify, unpack and measure time and ratio.

    Args:
        source: Source directory or file to benchmark.
        output: Output .kmc archive path.
        block_size: Block size in bytes.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()

    # Compute original size
    if source.is_file():
        orig_size = source.stat().st_size
    else:
        orig_size = sum(
            f.stat().st_size for f in source.rglob("*") if f.is_file()
        )

    print("=== KMC Benchmark ===")
    print(f"Source: {source}")
    print(f"Original size: {orig_size:,} bytes")
    print()

    # Pack
    print("Packing ...")
    t0 = time.time()
    pack(source, output, block_size=block_size)
    pack_time = time.time() - t0
    comp_size = output.stat().st_size
    ratio = comp_size / orig_size if orig_size > 0 else 0
    print(f"  Time: {pack_time:.3f}s")
    print(f"  Compressed size: {comp_size:,} bytes")
    print(f"  Ratio: {ratio:.2%}")
    print()

    # Verify
    print("Verifying ...")
    t0 = time.time()
    errors = verify(output)
    verify_time = time.time() - t0
    if errors:
        print(f"  FAILED — {len(errors)} error(s)")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"  OK — {verify_time:.3f}s")
    print()

    # Unpack
    with tempfile.TemporaryDirectory() as tmpdir:
        restore_dir = Path(tmpdir) / "restored"
        print("Unpacking ...")
        t0 = time.time()
        unpack(output, restore_dir)
        unpack_time = time.time() - t0
        print(f"  Time: {unpack_time:.3f}s")
        print()

    # Summary
    print("=== Summary ===")
    print(f"  Original size:   {orig_size:>12,} bytes")
    print(f"  Compressed size: {comp_size:>12,} bytes")
    print(f"  Ratio:           {ratio:>12.2%}")
    print(f"  Pack time:       {pack_time:>12.3f}s")
    print(f"  Verify time:     {verify_time:>12.3f}s")
    print(f"  Unpack time:     {unpack_time:>12.3f}s")
    throughput = orig_size / pack_time if pack_time > 0 else 0
    print(f"  Pack throughput: {throughput:>12,.0f} bytes/s")
