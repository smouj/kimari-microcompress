"""Command-line interface for Kimari MicroCompress."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, inspect, pack, unpack, verify_full
from .benchmark import (
    benchmark_to_json,
    format_benchmark_table,
    run_benchmark,
)
from .inspector import inspect_directory, inspect_file


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack a directory or file into a .kmc archive."""
    source = Path(args.source)
    output = Path(args.output)
    block_size = args.block_size or DEFAULT_BLOCK_SIZE
    level = args.level
    tensor_aware = getattr(args, "tensor_aware", False)

    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    mode_str = " (tensor-aware)" if tensor_aware else ""
    print(f"Packing {source} -> {output}{mode_str} (block_size={block_size}, level={level})")
    start = time.time()
    pack(source, output, block_size=block_size, level=level, tensor_aware=tensor_aware)
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
    """Verify the integrity of a .kmc archive with detailed report."""
    archive = Path(args.archive)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {archive} ...")
    report = verify_full(archive)
    print()
    print(report)

    if report.integrity != "OK":
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a .kmc archive or a directory/file for AI model formats."""
    target = Path(args.target)

    if not target.exists():
        print(f"Error: target not found: {target}", file=sys.stderr)
        sys.exit(1)

    show_tensors = getattr(args, "tensors", False)
    json_output = getattr(args, "json", False)

    # If it's a .kmc archive, show archive manifest
    if target.is_file() and target.suffix.lower() == ".kmc":
        _inspect_archive(target, json_output=json_output, show_tensors=show_tensors)
    else:
        _inspect_model(target, json_output=json_output, show_tensors=show_tensors)


def _format_size(n: int) -> str:
    """Format a byte count as a human-readable size string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} bytes"


def _inspect_archive(archive: Path, json_output: bool = False, show_tensors: bool = False) -> None:
    """Display archive manifest information."""
    manifest = inspect(archive)

    if json_output:
        data = {
            "type": "kmc_archive",
            "path": str(archive),
            "version": manifest.version,
            "tool": manifest.tool,
            "tool_version": manifest.tool_version,
            "created_at": manifest.created_at,
            "original_size": manifest.total_original_size,
            "compressed_size": manifest.total_compressed_size,
            "ratio": (
                manifest.total_compressed_size / manifest.total_original_size
                if manifest.total_original_size > 0
                else 0
            ),
            "files": [],
        }
        for fentry in manifest.files:
            file_data = {
                "path": fentry.path,
                "original_size": fentry.original_size,
                "hash": fentry.hash,
                "blocks": len(fentry.blocks),
                "block_size": fentry.block_size,
                "codecs": sorted(set(b.codec for b in fentry.blocks)),
                "tensor_count": fentry.tensor_count,
                "dtype_summary": fentry.dtype_summary,
            }
            if show_tensors and fentry.tensor_entries:
                file_data["tensor_entries"] = [
                    {
                        "name": t.name,
                        "dtype": t.dtype,
                        "shape": t.shape,
                        "byte_offset": t.byte_offset,
                        "byte_size": t.byte_size,
                    }
                    for t in fentry.tensor_entries
                ]
            data["files"].append(file_data)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    print(f"KMC Archive: {archive}")
    print(f"  Version: {manifest.version}")
    print(f"  Tool: {manifest.tool} v{manifest.tool_version}")
    print(f"  Created: {manifest.created_at}")
    print(f"  Original size: {_format_size(manifest.total_original_size)}")
    print(f"  Compressed size: {_format_size(manifest.total_compressed_size)}")

    if manifest.total_original_size > 0:
        ratio = manifest.total_compressed_size / manifest.total_original_size
    else:
        ratio = 0
    print(f"  Ratio: {ratio:.2%}")
    print(f"  Files: {len(manifest.files)}")
    print()

    for fentry in manifest.files:
        print(f"  {fentry.path}")
        print(f"    Size: {_format_size(fentry.original_size)} | Hash: {fentry.hash[:16]}...")
        print(f"    Blocks: {len(fentry.blocks)} (block_size={fentry.block_size:,})")
        codecs_used = set(b.codec for b in fentry.blocks)
        print(f"    Codecs: {', '.join(sorted(codecs_used))}")

        # Show tensor-aware info if available
        if fentry.tensor_count > 0:
            print(f"    Tensors: {fentry.tensor_count}")
            print(f"    Dtypes: {', '.join(fentry.dtype_summary)}")

        if show_tensors and fentry.tensor_entries:
            shown = min(10, len(fentry.tensor_entries))
            for t in fentry.tensor_entries[:shown]:
                shape_str = "x".join(str(d) for d in t.shape)
                print(f"      {t.name}: {t.dtype} [{shape_str}] ({_format_size(t.byte_size)})")
            if len(fentry.tensor_entries) > shown:
                print(f"      ... and {len(fentry.tensor_entries) - shown} more tensors")


def _inspect_model(target: Path, json_output: bool = False, show_tensors: bool = False) -> None:
    """Display AI model format information for a file or directory."""
    if target.is_file():
        results = [inspect_file(target)]
    else:
        results = inspect_directory(target)

    # Also get directory-level info if it's a directory
    dir_info = None
    if target.is_dir():
        dir_info = _get_directory_model_info(target)

    if json_output:
        data = {
            "type": "model_directory" if target.is_dir() else "model_file",
            "path": str(target),
        }
        if dir_info:
            data.update(dir_info)
        data["files"] = []
        for r in results:
            file_data = {
                "path": str(r.path.relative_to(target)) if target.is_dir() else r.path.name,
                "format": r.format.value,
                "details": r.details,
                "file_size": r.file_size,
            }
            if r.tensors:
                file_data["tensors"] = [
                    {
                        "name": t.name,
                        "dtype": t.dtype,
                        "shape": t.shape,
                        "byte_offset": t.byte_offset,
                        "byte_size": t.byte_size,
                    }
                    for t in r.tensors
                ]
            if r.extra:
                file_data["extra"] = {k: v for k, v in r.extra.items() if k != "header_size"}
            data["files"].append(file_data)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    print("KMC Model Inspection")
    print()
    print(f"Path: {target}")

    if dir_info:
        print(f"Detected type: {dir_info.get('detected_type', 'unknown')}")
        print(f"Safetensors: {'yes' if dir_info.get('has_safetensors') else 'no'}")
        print(f"Sharded: {'yes' if dir_info.get('is_sharded') else 'no'}")

        tensor_count = dir_info.get("total_tensor_count", 0)
        tensor_bytes = dir_info.get("total_tensor_bytes", 0)
        if tensor_count > 0:
            print(f"Tensor count: {tensor_count}")
            print(f"Total tensor bytes: {_format_size(tensor_bytes)}")

        dtypes = dir_info.get("dtypes", {})
        if dtypes:
            print("Dtypes:")
            for dtype, count in sorted(dtypes.items()):
                print(f"  {dtype}: {count} tensors")

        print(f"LoRA/PEFT: {'yes' if dir_info.get('is_lora') else 'no'}")
        if dir_info.get("is_lora"):
            lora_info = dir_info.get("lora_info", {})
            if lora_info.get("lora_rank"):
                print(f"  Rank: {lora_info['lora_rank']}")
            if lora_info.get("target_modules"):
                print(f"  Target modules: {', '.join(lora_info['target_modules'])}")
            if lora_info.get("base_model_reference"):
                print(f"  Base model: {lora_info['base_model_reference']}")

        print(f"GGUF: {'yes' if dir_info.get('has_gguf') else 'no'}")

        file_list = dir_info.get("model_files", [])
        if file_list:
            print()
            print("Files:")
            for fn in file_list:
                print(f"  {fn}")
    else:
        # Single file
        if results:
            r = results[0]
            print(f"Detected type: {r.format.value}")
            print(f"File size: {_format_size(r.file_size)}")
            print(f"Details: {r.details}")

    print()

    # Details per file
    if not dir_info or show_tensors:
        for r in results:
            rel = r.path.name if r.path.parent == target else str(r.path.relative_to(target))
            size_str = f" ({_format_size(r.file_size)})" if r.file_size else ""
            print(f"  {rel} [{r.format.value}]{size_str}")
            print(f"    {r.details}")

            # Show tensor info for safetensors
            if r.tensors and show_tensors:
                shown = min(10, len(r.tensors))
                for t in r.tensors[:shown]:
                    shape_str = "x".join(str(d) for d in t.shape)
                    print(f"      {t.name}: {t.dtype} [{shape_str}] ({_format_size(t.byte_size)})")
                if len(r.tensors) > shown:
                    print(f"      ... and {len(r.tensors) - shown} more tensors")

            # Show extra metadata
            if r.extra:
                for key, value in r.extra.items():
                    if key not in ("header_size",):
                        print(f"      {key}: {value}")


def _get_directory_model_info(directory: Path) -> dict:
    """Analyze a directory for model-level information.

    Returns a dict with high-level model detection results including
    safetensors status, sharding, LoRA detection, GGUF detection,
    tensor counts, and dtype summaries.
    """
    from .formats.safetensors import detect_lora_adapter
    from .inspector import ModelFormat

    results = inspect_directory(directory)

    has_safetensors = False
    has_gguf = False
    is_sharded = False
    total_tensor_count = 0
    total_tensor_bytes = 0
    dtypes: dict[str, int] = {}
    model_files: list[str] = []

    for r in results:
        if r.format == ModelFormat.SAFETENSORS or r.format == ModelFormat.LORA_ADAPTER:
            has_safetensors = True
            if r.tensors:
                total_tensor_count += len(r.tensors)
                for t in r.tensors:
                    total_tensor_bytes += t.byte_size
                    dtypes[t.dtype] = dtypes.get(t.dtype, 0) + 1
        if r.format == ModelFormat.GGUF:
            has_gguf = True
        if r.format == ModelFormat.SHARD:
            is_sharded = True

        rel = str(r.path.relative_to(directory))
        model_files.append(rel)

    # Check for shard index file
    shard_index = directory / "model.safetensors.index.json"
    if shard_index.is_file():
        is_sharded = True

    # Check for GGUF files
    for f in directory.rglob("*.gguf"):
        if f.is_file():
            has_gguf = True

    # Detect LoRA
    lora_info = detect_lora_adapter(directory)
    is_lora = lora_info.get("is_lora", False)

    # Detect detected type
    if is_lora:
        detected_type = "PEFT/LoRA adapter"
    elif has_safetensors and has_gguf:
        detected_type = "Mixed format model"
    elif has_safetensors:
        detected_type = "Hugging Face model folder"
    elif has_gguf:
        detected_type = "GGUF model"
    else:
        detected_type = "unknown"

    return {
        "detected_type": detected_type,
        "has_safetensors": has_safetensors,
        "has_gguf": has_gguf,
        "is_sharded": is_sharded,
        "total_tensor_count": total_tensor_count,
        "total_tensor_bytes": total_tensor_bytes,
        "dtypes": dtypes,
        "is_lora": is_lora,
        "lora_info": lora_info,
        "model_files": model_files,
    }


def cmd_bench(args: argparse.Namespace) -> None:
    """Run a benchmark: pack, verify, unpack and measure performance."""
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    compare_zipnn = getattr(args, "compare_zipnn", False)
    tensor_aware = getattr(args, "tensor_aware", False)

    result = run_benchmark(
        source,
        output,
        synthetic=getattr(args, "synthetic", False),
        tensor_aware=tensor_aware,
        compare_zipnn=compare_zipnn,
    )

    if args.json:
        json_output = benchmark_to_json(result)
        if args.output_file:
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_file).write_text(json_output)
            print(f"Benchmark results saved to {args.output_file}")
        else:
            print(json_output)
    else:
        print(format_benchmark_table(result))


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
    p_pack.add_argument("-b", "--block-size", type=int, default=None, help="Block size in bytes")
    p_pack.add_argument("-l", "--level", type=int, default=3, help="Compression level")
    p_pack.add_argument(
        "--tensor-aware",
        action="store_true",
        help="Align blocks to tensor boundaries for safetensors files",
    )
    p_pack.set_defaults(func=cmd_pack)

    # unpack
    p_unpack = sub.add_parser("unpack", help="Unpack a .kmc archive")
    p_unpack.add_argument("archive", help=".kmc archive path")
    p_unpack.add_argument("output", help="Output directory")
    p_unpack.set_defaults(func=cmd_unpack)

    # verify
    p_verify = sub.add_parser("verify", help="Verify a .kmc archive integrity (full report)")
    p_verify.add_argument("archive", help=".kmc archive path")
    p_verify.set_defaults(func=cmd_verify)

    # inspect
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect a .kmc archive or AI model directory/file",
    )
    p_inspect.add_argument("target", help=".kmc archive path or directory/file to inspect")
    p_inspect.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    p_inspect.add_argument(
        "--tensors",
        action="store_true",
        help="Show detailed tensor information",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark pack/verify/unpack")
    p_bench.add_argument("source", help="Source directory or file")
    p_bench.add_argument("output", help="Output .kmc archive path for benchmark")
    p_bench.add_argument("--json", action="store_true", help="Output results as JSON")
    p_bench.add_argument("--output-file", default=None, help="Save benchmark results to file")
    p_bench.add_argument("--synthetic", action="store_true", help="Mark data as synthetic")
    p_bench.add_argument(
        "--tensor-aware",
        action="store_true",
        help="Use tensor-aware compression mode",
    )
    p_bench.add_argument(
        "--compare-zipnn",
        action="store_true",
        help="Compare with ZipNN if available",
    )
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
