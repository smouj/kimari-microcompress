"""Command-line interface for Kimari MicroCompress."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, inspect, pack, unpack, verify
from .benchmark import run_benchmark


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack a directory or file into a .kmc archive."""
    source = Path(args.source)
    output = Path(args.output)
    block_size = args.block_size or DEFAULT_BLOCK_SIZE
    level = args.level

    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Packing {source} -> {output} (block_size={block_size}, level={level})")
    start = time.time()
    pack(source, output, block_size=block_size, level=level)
    elapsed = time.time() - start

    orig = (
        source.stat().st_size
        if source.is_file()
        else sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    )
    comp = output.stat().st_size
    ratio = comp / orig if orig > 0 else 0

    print(f"Done in {elapsed:.2f}s — {orig:,} -> {comp:,} bytes (ratio: {ratio:.2%})")


def cmd_unpack(args: argparse.Namespace) -> None:
    """Unpack a .kmc archive to a directory."""
    archive = Path(args.archive)
    output_dir = Path(args.output)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    print(f"Unpacking {archive} -> {output_dir}")
    start = time.time()
    unpack(archive, output_dir)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify the integrity of a .kmc archive."""
    archive = Path(args.archive)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {archive} ...")
    errors = verify(archive)

    if errors:
        print(f"FAILED — {len(errors)} error(s) found:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK — all blocks and hashes verified successfully.")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a .kmc archive and display its manifest."""
    archive = Path(args.archive)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    manifest = inspect(archive)
    print(f"KMC Archive: {archive}")
    print(f"  Version: {manifest.version}")
    print(f"  Tool: {manifest.tool} v{manifest.tool_version}")
    print(f"  Created: {manifest.created_at}")
    print(f"  Original size: {manifest.total_original_size:,} bytes")
    print(f"  Compressed size: {manifest.total_compressed_size:,} bytes")

    if manifest.total_original_size > 0:
        ratio = manifest.total_compressed_size / manifest.total_original_size
    else:
        ratio = 0
    print(f"  Ratio: {ratio:.2%}")
    print(f"  Files: {len(manifest.files)}")
    print()

    for fentry in manifest.files:
        print(f"  {fentry.path}")
        print(f"    Size: {fentry.original_size:,} bytes | Hash: {fentry.hash[:16]}...")
        print(f"    Blocks: {len(fentry.blocks)} (block_size={fentry.block_size:,})")
        codecs_used = set(b.codec for b in fentry.blocks)
        print(f"    Codecs: {', '.join(sorted(codecs_used))}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run a benchmark: pack, verify, unpack and measure performance."""
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    run_benchmark(source, output)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the KMC CLI."""
    parser = argparse.ArgumentParser(
        prog="kmc",
        description="Kimari MicroCompress — reversible lossless compression for AI models",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # pack
    p_pack = sub.add_parser("pack", help="Pack a directory/file into a .kmc archive")
    p_pack.add_argument("source", help="Source directory or file")
    p_pack.add_argument("output", help="Output .kmc archive path")
    p_pack.add_argument(
        "-b", "--block-size", type=int, default=None, help="Block size in bytes"
    )
    p_pack.add_argument(
        "-l", "--level", type=int, default=3, help="Compression level"
    )
    p_pack.set_defaults(func=cmd_pack)

    # unpack
    p_unpack = sub.add_parser("unpack", help="Unpack a .kmc archive")
    p_unpack.add_argument("archive", help=".kmc archive path")
    p_unpack.add_argument("output", help="Output directory")
    p_unpack.set_defaults(func=cmd_unpack)

    # verify
    p_verify = sub.add_parser("verify", help="Verify a .kmc archive integrity")
    p_verify.add_argument("archive", help=".kmc archive path")
    p_verify.set_defaults(func=cmd_verify)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect a .kmc archive manifest")
    p_inspect.add_argument("archive", help=".kmc archive path")
    p_inspect.set_defaults(func=cmd_inspect)

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark pack/verify/unpack")
    p_bench.add_argument("source", help="Source directory or file")
    p_bench.add_argument("output", help="Output .kmc archive path for benchmark")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main() -> None:
    """Entry point for the KMC CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
