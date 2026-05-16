"""Kimari CLI adapter example: demonstrates how to integrate KMC with Kimari.

This example shows how the Kimari CLI can use the KMC integration layer
to provide compression, decompression, verification, benchmark,
LoRA, and checkpoint commands without directly depending on KMC internals.

Usage:
    python examples/kimari_cli_adapter.py compress ./model ./model.kmc
    python examples/kimari_cli_adapter.py compress-lora ./adapter ./adapter.kmc
    python examples/kimari_cli_adapter.py compress-checkpoint ./ckpt ./ckpt.kmc
    python examples/kimari_cli_adapter.py decompress ./model.kmc ./restored
    python examples/kimari_cli_adapter.py verify ./model.kmc
    python examples/kimari_cli_adapter.py bench ./model ./model.kmc
    python examples/kimari_cli_adapter.py inspect-model ./model
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
        gguf_aware=getattr(args, "gguf_aware", False),
    )
    print(json.dumps(result, indent=2))


def cmd_compress_lora(args: argparse.Namespace) -> None:
    """Kimari compress-lora command."""
    from kmc.integrations.kimari import kimari_pack_lora

    result = kimari_pack_lora(
        input_path=args.source,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))

    if result.get("status") == "error":
        sys.exit(1)


def cmd_compress_checkpoint(args: argparse.Namespace) -> None:
    """Kimari compress-checkpoint command."""
    from kmc.integrations.kimari import kimari_pack_checkpoint

    result = kimari_pack_checkpoint(
        input_path=args.source,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))

    if result.get("status") == "error":
        sys.exit(1)


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


def cmd_inspect_model(args: argparse.Namespace) -> None:
    """Kimari inspect-model command."""
    from kmc.integrations.kimari import kimari_inspect_model

    result = kimari_inspect_model(
        input_path=args.source,
        json_output=True,
    )
    print(json.dumps(result, indent=2))


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
    p_compress.add_argument(
        "--gguf-aware",
        action="store_true",
        help="Enable experimental GGUF-aware mode",
    )
    p_compress.set_defaults(func=cmd_compress)

    # compress-lora
    p_lora = sub.add_parser("compress-lora", help="Compress a LoRA adapter")
    p_lora.add_argument("source", help="LoRA adapter directory")
    p_lora.add_argument("output", help="Output .kmc archive path")
    p_lora.set_defaults(func=cmd_compress_lora)

    # compress-checkpoint
    p_ckpt = sub.add_parser("compress-checkpoint", help="Compress a training checkpoint")
    p_ckpt.add_argument("source", help="Checkpoint directory")
    p_ckpt.add_argument("output", help="Output .kmc archive path")
    p_ckpt.set_defaults(func=cmd_compress_checkpoint)

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

    # inspect-model
    p_inspect = sub.add_parser("inspect-model", help="Inspect a model")
    p_inspect.add_argument("source", help="Model directory or file")
    p_inspect.set_defaults(func=cmd_inspect_model)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
