"""Kimari CLI adapter example: demonstrates how to integrate KMC with Kimari.

This example shows how the Kimari CLI can use the KMC integration layer
to provide compression, decompression, verification, and benchmark
commands without directly depending on KMC internals.

Usage:
    python examples/kimari_cli_adapter.py compress ./model ./model.kmc
    python examples/kimari_cli_adapter.py decompress ./model.kmc ./restored
    python examples/kimari_cli_adapter.py verify ./model.kmc
    python examples/kimari_cli_adapter.py bench ./model ./model.kmc --compare-zipnn
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_compress(args: argparse.Namespace) -> None:
    """Kimari compress command."""
    from kmc.integrations.kimari import kimari_compress

    result = kimari_compress(
        source=args.source,
        output=args.output,
        tensor_aware=getattr(args, "tensor_aware", True),
    )
    print(json.dumps(result, indent=2))


def cmd_decompress(args: argparse.Namespace) -> None:
    """Kimari decompress command."""
    from kmc.integrations.kimari import kimari_decompress

    result = kimari_decompress(
        archive=args.archive,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2))


def cmd_verify_compress(args: argparse.Namespace) -> None:
    """Kimari verify-compress command."""
    from kmc.integrations.kimari import kimari_verify_compress

    result = kimari_verify_compress(archive=args.archive)
    print(json.dumps(result, indent=2))

    if result["status"] != "ok":
        sys.exit(1)


def cmd_bench_compress(args: argparse.Namespace) -> None:
    """Kimari bench-compress command."""
    from kmc.benchmark import format_benchmark_table
    from kmc.integrations.kimari import kimari_bench_compress

    result = kimari_bench_compress(
        source=args.source,
        output=args.output,
        tensor_aware=getattr(args, "tensor_aware", True),
        compare_zipnn=getattr(args, "compare_zipnn", False),
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_benchmark_table(result))


def main() -> None:
    """Entry point for the Kimari CLI adapter example."""
    parser = argparse.ArgumentParser(
        prog="kimari-cli-adapter",
        description="Kimari CLI adapter for KMC compression operations",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # compress
    p_compress = sub.add_parser("compress", help="Compress a model")
    p_compress.add_argument("source", help="Source directory or file")
    p_compress.add_argument("output", help="Output .kmc archive path")
    p_compress.add_argument(
        "--no-tensor-aware",
        dest="tensor_aware",
        action="store_false",
        help="Disable tensor-aware mode",
    )
    p_compress.set_defaults(func=cmd_compress)

    # decompress
    p_decompress = sub.add_parser("decompress", help="Decompress a .kmc archive")
    p_decompress.add_argument("archive", help=".kmc archive path")
    p_decompress.add_argument("output", help="Output directory")
    p_decompress.set_defaults(func=cmd_decompress)

    # verify-compress
    p_verify = sub.add_parser("verify-compress", help="Verify archive integrity")
    p_verify.add_argument("archive", help=".kmc archive path")
    p_verify.set_defaults(func=cmd_verify_compress)

    # bench-compress
    p_bench = sub.add_parser("bench-compress", help="Benchmark compression")
    p_bench.add_argument("source", help="Source directory or file")
    p_bench.add_argument("output", help="Output .kmc archive path")
    p_bench.add_argument("--json", action="store_true", help="Output as JSON")
    p_bench.add_argument(
        "--compare-zipnn",
        action="store_true",
        help="Compare with ZipNN if available",
    )
    p_bench.add_argument(
        "--no-tensor-aware",
        dest="tensor_aware",
        action="store_false",
        help="Disable tensor-aware mode",
    )
    p_bench.set_defaults(func=cmd_bench_compress)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
