"""Command-line interface for Kimari MicroCompress."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .archive import DEFAULT_BLOCK_SIZE, inspect, pack, unpack, verify_full, verify_quick
from .benchmark import (
    benchmark_to_json,
    format_benchmark_table,
    run_benchmark,
)
from .inspector import inspect_directory, inspect_file
from .reader import KMCReader


def cmd_pack(args: argparse.Namespace) -> None:
    """Pack a directory or file into a .kmc archive."""
    source = Path(args.source)
    output = Path(args.output)
    block_size = args.block_size or DEFAULT_BLOCK_SIZE
    level = args.level
    tensor_aware = getattr(args, "tensor_aware", False)
    codec = getattr(args, "codec", "auto")
    gguf_aware = getattr(args, "gguf_aware", False)
    jobs = getattr(args, "jobs", 1)
    show_progress = getattr(args, "progress", False)
    dedup = getattr(args, "dedup", False)
    delta_base = getattr(args, "delta_base", None)

    if not source.exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    # Validate codec choice
    valid_codecs = {"auto", "raw", "zlib", "zstd", "byteplane", "floatplane", "gguf_quant_block"}
    if codec not in valid_codecs:
        print(f"Error: unknown codec '{codec}'. Valid: {sorted(valid_codecs)}", file=sys.stderr)
        sys.exit(1)

    mode_str = " (tensor-aware)" if tensor_aware else ""
    codec_str = f" --codec {codec}" if codec != "auto" else ""
    gguf_str = " (gguf-aware)" if gguf_aware else ""
    jobs_str = f" --jobs {jobs}" if jobs > 1 else ""
    dedup_str = " --dedup" if dedup else ""
    delta_str = f" --delta-base {delta_base}" if delta_base else ""
    extras = f"{mode_str}{codec_str}{gguf_str}{jobs_str}{dedup_str}{delta_str}"
    print(f"Packing {source} -> {output}{extras} (block_size={block_size}, level={level})")

    from .reporting import create_reporter

    reporter = create_reporter(show_progress=show_progress, json_mode=False)
    reporter.start("Packing")

    start = time.time()
    pack(
        source,
        output,
        block_size=block_size,
        level=level,
        tensor_aware=tensor_aware,
        codec=codec,
        gguf_aware=gguf_aware,
        jobs=jobs,
        progress_reporter=reporter,
        dedup=dedup,
        delta_base=Path(delta_base) if delta_base else None,
    )
    elapsed = time.time() - start

    orig = (
        source.stat().st_size
        if source.is_file()
        else sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    )
    comp = output.stat().st_size
    ratio = comp / orig if orig > 0 else 0

    reporter.finish(f"{elapsed:.2f}s")
    print(f"Done in {elapsed:.2f}s — {orig:,} -> {comp:,} bytes (ratio: {ratio:.2%})")


def cmd_pack_lora(args: argparse.Namespace) -> None:
    """Pack a LoRA adapter directory into a .kmc archive."""
    source = Path(args.source)
    output = Path(args.output)
    block_size = args.block_size or DEFAULT_BLOCK_SIZE
    level = args.level
    codec = getattr(args, "codec", "auto")

    if not source.is_dir():
        print(f"Error: source must be a directory: {source}", file=sys.stderr)
        sys.exit(1)

    from .workflows.lora import build_lora_manifest_metadata, detect_lora_adapter

    adapter_info = detect_lora_adapter(source)
    if not adapter_info.is_lora:
        print(f"Error: not a LoRA adapter directory: {source}", file=sys.stderr)
        sys.exit(1)

    if adapter_info.warnings:
        for w in adapter_info.warnings:
            print(f"Warning: {w}", file=sys.stderr)

    artifact_metadata = build_lora_manifest_metadata(adapter_info)
    print(f"Packing LoRA adapter: {source} -> {output}")
    print(f"  PEFT type: {adapter_info.peft_type}")
    if adapter_info.lora_rank is not None:
        print(f"  Rank: {adapter_info.lora_rank}")
    if adapter_info.target_modules:
        print(f"  Target modules: {', '.join(adapter_info.target_modules)}")
    if adapter_info.base_model_name_or_path != "unknown":
        print(f"  Base model: {adapter_info.base_model_name_or_path}")

    start = time.time()
    pack(
        source,
        output,
        block_size=block_size,
        level=level,
        tensor_aware=True,
        codec=codec,
        artifact_type="lora_adapter",
        artifact_metadata=artifact_metadata,
    )
    elapsed = time.time() - start

    orig = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    comp = output.stat().st_size
    ratio = comp / orig if orig > 0 else 0

    print(f"Done in {elapsed:.2f}s — {orig:,} -> {comp:,} bytes (ratio: {ratio:.2%})")


def cmd_pack_checkpoint(args: argparse.Namespace) -> None:
    """Pack a training checkpoint directory into a .kmc archive."""
    source = Path(args.source)
    output = Path(args.output)
    block_size = args.block_size or DEFAULT_BLOCK_SIZE
    level = args.level
    codec = getattr(args, "codec", "auto")

    if not source.is_dir():
        print(f"Error: source must be a directory: {source}", file=sys.stderr)
        sys.exit(1)

    from .workflows.checkpoint import build_checkpoint_manifest_metadata, detect_checkpoint

    ckpt_info = detect_checkpoint(source)
    if not ckpt_info.is_checkpoint:
        print(f"Error: not a training checkpoint directory: {source}", file=sys.stderr)
        sys.exit(1)

    if ckpt_info.warnings:
        for w in ckpt_info.warnings:
            print(f"Warning: {w}", file=sys.stderr)

    artifact_metadata = build_checkpoint_manifest_metadata(ckpt_info)
    print(f"Packing training checkpoint: {source} -> {output}")
    if ckpt_info.step is not None:
        print(f"  Step: {ckpt_info.step}")
    print(f"  Has optimizer state: {ckpt_info.has_optimizer_state}")
    print(f"  Has trainer state: {ckpt_info.has_trainer_state}")
    print(f"  Has safetensors model: {ckpt_info.has_safetensors_model}")

    tensor_aware = ckpt_info.has_safetensors_model
    start = time.time()
    pack(
        source,
        output,
        block_size=block_size,
        level=level,
        tensor_aware=tensor_aware,
        codec=codec,
        artifact_type="training_checkpoint",
        artifact_metadata=artifact_metadata,
    )
    elapsed = time.time() - start

    orig = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
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

    # Selective extraction flags
    only_patterns = getattr(args, "only", None)
    tensor_name = getattr(args, "tensor", None)
    list_only = getattr(args, "list", False)
    json_output = getattr(args, "json", False)

    # --list mode: just list available files and exit
    if list_only:
        try:
            with KMCReader(archive) as reader:
                files = reader.list_files()
                tensors = reader.list_tensors()
                if json_output:
                    print(json.dumps({"files": files, "tensors": tensors}, indent=2))
                else:
                    print("Available files:")
                    for f in files:
                        print(f"  {f}")
                    if tensors:
                        print("Available tensors:")
                        for t in tensors:
                            print(f"  {t}")
        except (ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Selective extraction via --only or --tensor
    if only_patterns or tensor_name:
        try:
            with KMCReader(archive) as reader:
                if only_patterns:
                    _selective_unpack_files(reader, only_patterns, output_dir, json_output)
                elif tensor_name:
                    _selective_unpack_tensor(reader, tensor_name, output_dir, json_output)
        except (ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Full unpack (default behavior)
    print(f"Unpacking {archive} -> {output_dir}")
    start = time.time()
    unpack(archive, output_dir)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify the integrity of a .kmc archive with detailed report."""
    archive = Path(args.archive)
    verify_mode = getattr(args, "verify_mode", "full")

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    if verify_mode == "quick":
        print(f"Quick-verifying {archive} ...")
        report = verify_quick(archive)
    else:
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
    show_compression = getattr(args, "compression", False)
    show_lora = getattr(args, "lora", False)
    show_checkpoint = getattr(args, "checkpoint", False)
    show_gguf = getattr(args, "gguf", False)
    show_dedup = getattr(args, "dedup", False)
    show_delta = getattr(args, "delta_info", False)
    show_runtime_hints = getattr(args, "runtime_hints", False)

    # If it's a .kmc archive, show archive manifest
    if target.is_file() and target.suffix.lower() == ".kmc":
        _inspect_archive(
            target,
            json_output=json_output,
            show_tensors=show_tensors,
            show_compression=show_compression,
            show_dedup=show_dedup,
            show_delta=show_delta,
            show_runtime_hints=show_runtime_hints,
        )
    else:
        _inspect_model(
            target,
            json_output=json_output,
            show_tensors=show_tensors,
            show_lora=show_lora,
            show_checkpoint=show_checkpoint,
            show_gguf=show_gguf,
        )


def cmd_list(args: argparse.Namespace) -> None:
    """List the contents of a .kmc archive."""
    archive = Path(args.archive)
    show_files = getattr(args, "files", False)
    show_tensors = getattr(args, "tensors", False)
    json_output = getattr(args, "json", False)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    try:
        with KMCReader(archive) as reader:
            files = reader.list_files()
            tensors = reader.list_tensors()
            manifest = reader.get_manifest()

            if json_output:
                data: dict = {
                    "archive": str(archive),
                    "version": manifest.version,
                    "files": [],
                    "tensors": [],
                }
                for f_path in files:
                    info = reader.get_file_info(f_path)
                    entry = {"path": f_path}
                    if info:
                        entry["size"] = info.size
                        entry["sha256"] = info.sha256
                    data["files"].append(entry)
                for t_name in tensors:
                    info = reader.get_tensor_info(t_name)
                    entry: dict = {"name": t_name}
                    if info:
                        entry["file_path"] = info.file_path
                        entry["dtype"] = info.dtype
                        entry["shape"] = info.shape
                    data["tensors"].append(entry)
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return

            # Human-readable output
            print("KMC Archive Contents")
            print()

            # Show files
            if show_tensors and not show_files:
                # Only show tensors
                pass
            else:
                print("Files:")
                for f_path in files:
                    info = reader.get_file_info(f_path)
                    size_str = f" ({_format_size(info.size)})" if info and info.size else ""
                    print(f"  {f_path}{size_str}")
                print()

            # Show tensors
            if show_files and not show_tensors:
                # Only show files
                pass
            else:
                if tensors:
                    print("Tensors:")
                    for t_name in tensors:
                        info = reader.get_tensor_info(t_name)
                        if info and info.dtype and info.shape:
                            shape_str = "x".join(str(d) for d in info.shape)
                            print(f"  {t_name}  {info.dtype}  [{shape_str}]")
                        else:
                            print(f"  {t_name}")
                else:
                    if not show_files:
                        print("Tensors: (none — archive not created with --tensor-aware)")
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _format_size(n: int) -> str:
    """Format a byte count as a human-readable size string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} bytes"


def _inspect_archive(
    archive: Path,
    json_output: bool = False,
    show_tensors: bool = False,
    show_compression: bool = False,
    show_dedup: bool = False,
    show_delta: bool = False,
    show_runtime_hints: bool = False,
) -> None:
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
            "artifact_type": manifest.artifact_type,
            "original_size": manifest.total_original_size,
            "compressed_size": manifest.total_compressed_size,
            "ratio": (
                manifest.total_compressed_size / manifest.total_original_size
                if manifest.total_original_size > 0
                else 0
            ),
            "files": [],
        }
        if manifest.artifact_metadata:
            data["artifact_metadata"] = manifest.artifact_metadata
        if manifest.format_metadata:
            data["format_metadata"] = manifest.format_metadata
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
            if show_compression:
                file_data["compression_summary"] = _get_compression_summary(fentry)
            data["files"].append(file_data)

        # Add overall compression summary
        if show_compression:
            data["compression_summary"] = _get_overall_compression_summary(manifest)

        if manifest.deduplication:
            data["deduplication"] = manifest.deduplication
        if manifest.delta:
            data["delta"] = manifest.delta
        if manifest.runtime_hints:
            data["runtime_hints"] = manifest.runtime_hints

        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    print(f"KMC Archive: {archive}")
    print(f"  Version: {manifest.version}")
    print(f"  Tool: {manifest.tool} v{manifest.tool_version}")
    print(f"  Created: {manifest.created_at}")
    if manifest.artifact_type and manifest.artifact_type != "unknown":
        print(f"  Artifact type: {manifest.artifact_type}")
    print(f"  Original size: {_format_size(manifest.total_original_size)}")
    print(f"  Compressed size: {_format_size(manifest.total_compressed_size)}")

    if manifest.total_original_size > 0:
        ratio = manifest.total_compressed_size / manifest.total_original_size
    else:
        ratio = 0
    print(f"  Ratio: {ratio:.2%}")
    print(f"  Files: {len(manifest.files)}")
    print()

    # Show artifact metadata if present
    if manifest.artifact_metadata:
        print("Artifact metadata:")
        for key, value in manifest.artifact_metadata.items():
            print(f"  {key}: {value}")
        print()

    # Show format metadata if present
    if manifest.format_metadata:
        print("Format metadata:")
        for fmt_name, fmt_data in manifest.format_metadata.items():
            print(f"  {fmt_name}:")
            if isinstance(fmt_data, dict):
                for k, v in fmt_data.items():
                    print(f"    {k}: {v}")
        print()

    # Show compression summary if requested
    if show_compression:
        summary = _get_overall_compression_summary(manifest)
        print("Compression summary:")
        print(f"  Files: {summary.get('files', 0)}")
        print(f"  Blocks: {summary.get('blocks', 0)}")
        codec_usage = summary.get("codec_usage", {})
        if codec_usage:
            print("  Codec usage:")
            for codec_name, count in sorted(codec_usage.items()):
                print(f"    {codec_name}: {count} blocks")
        tensor_aware = summary.get("tensor_aware", False)
        if tensor_aware:
            print("  Tensor-aware: yes")
            tensor_dtypes = summary.get("tensor_dtypes", {})
            if tensor_dtypes:
                print("  Tensor dtypes:")
                for dtype, count in sorted(tensor_dtypes.items()):
                    print(f"    {dtype}: {count} tensors")
        else:
            print("  Tensor-aware: no")
        print()

    # Show partial access info (v0.7+)
    has_block_offsets = manifest.index.get("has_block_offsets", False) if manifest.index else False
    has_file_index = manifest.index.get("has_file_index", False) if manifest.index else False
    has_tensor_index = manifest.index.get("has_tensor_index", False) if manifest.index else False

    # Even without explicit index metadata, we can determine capabilities
    if not manifest.index:
        # Reconstruct from manifest content
        has_block_offsets = any(b.archive_offset > 0 for f in manifest.files for b in f.blocks)
        has_file_index = len(manifest.files) > 0
        has_tensor_index = any(b.tensor_name for f in manifest.files for b in f.blocks)

    tensor_status = "supported" if has_tensor_index else "unavailable"

    print("Partial access:")
    print(f"  Block index: {'yes' if has_block_offsets else 'reconstructed'}")
    print(f"  File index: {'yes' if has_file_index else 'no'}")
    print(f"  Tensor index: {'yes' if has_tensor_index else 'unavailable'}")
    print("  Selective extraction: supported")
    print(f"  Tensor extraction: {tensor_status}")
    print()

    # Show deduplication info (v0.8+)
    if show_dedup or manifest.deduplication:
        dedup = manifest.deduplication
        print("Deduplication:")
        if dedup.get("enabled"):
            print("  Enabled: yes")
            print(f"  Unique blocks: {dedup.get('unique_blocks', 0)}")
            print(f"  Deduplicated blocks: {dedup.get('deduplicated_blocks', 0)}")
            saved = dedup.get("saved_bytes", 0)
            print(f"  Saved bytes: {_format_size(saved)}")
        else:
            print("  Enabled: no")
        print()

    # Show delta info (v0.8+)
    if show_delta or manifest.delta:
        delta = manifest.delta
        print("Delta:")
        if delta.get("enabled"):
            print("  Enabled: yes")
            print(f"  Mode: {delta.get('mode', 'unknown')}")
            print(f"  Base archive: {delta.get('base_archive_path_hint', 'unknown')}")
        else:
            print("  Enabled: no")
        print()

    # Show runtime hints (v0.8+)
    if show_runtime_hints or manifest.runtime_hints:
        hints = manifest.runtime_hints
        print("Runtime hints:")
        print(f"  Partial file access: {hints.get('partial_file_access', 'no')}")
        print(f"  Tensor access: {hints.get('tensor_access', 'none')}")
        print(f"  Compressed inference: {hints.get('compressed_inference', False)}")
        print()

    for fentry in manifest.files:
        print(f"  {fentry.path}")
        print(f"    Size: {_format_size(fentry.original_size)} | Hash: {fentry.hash[:16]}...")
        print(f"    Blocks: {len(fentry.blocks)} (block_size={fentry.block_size:,})")
        codecs_used = set(b.codec for b in fentry.blocks)
        print(f"    Codecs: {', '.join(sorted(codecs_used))}")

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


def _get_compression_summary(fentry: object) -> dict:
    """Get compression summary for a single file entry."""
    # fentry is a FileEntry
    codec_usage: dict[str, int] = {}
    total_blocks = len(fentry.blocks)  # type: ignore[attr-defined]
    for block in fentry.blocks:  # type: ignore[attr-defined]
        codec_usage[block.codec] = codec_usage.get(block.codec, 0) + 1

    return {
        "blocks": total_blocks,
        "codec_usage": codec_usage,
    }


def _get_overall_compression_summary(manifest: object) -> dict:
    """Get overall compression summary from manifest."""
    manifest = manifest  # type: ignore[assignment]
    codec_usage: dict[str, int] = {}
    total_blocks = 0
    total_files = len(manifest.files)  # type: ignore[attr-defined]
    tensor_dtypes: dict[str, int] = {}
    has_tensor_data = False

    for fentry in manifest.files:  # type: ignore[attr-defined]
        total_blocks += len(fentry.blocks)
        for block in fentry.blocks:
            codec_usage[block.codec] = codec_usage.get(block.codec, 0) + 1

        if fentry.tensor_count > 0:
            has_tensor_data = True
            for dtype in fentry.dtype_summary:
                tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + fentry.tensor_count

        # Also count from block-level tensor_dtype
        for block in fentry.blocks:
            if block.tensor_dtype:
                has_tensor_data = True
                tensor_dtypes[block.tensor_dtype] = tensor_dtypes.get(block.tensor_dtype, 0) + 1

    return {
        "files": total_files,
        "blocks": total_blocks,
        "codec_usage": codec_usage,
        "tensor_aware": has_tensor_data,
        "tensor_dtypes": tensor_dtypes if has_tensor_data else {},
    }


def _inspect_model(
    target: Path,
    json_output: bool = False,
    show_tensors: bool = False,
    show_lora: bool = False,
    show_checkpoint: bool = False,
    show_gguf: bool = False,
) -> None:
    """Display AI model format information for a file or directory."""
    if target.is_file():
        results = [inspect_file(target)]
    else:
        results = inspect_directory(target)

    # Detect artifact type
    artifact_type = "unknown"
    dir_info = None
    if target.is_dir():
        dir_info = _get_directory_model_info(target)
        artifact_type = dir_info.get("detected_type_artifact", "unknown")

    # Check for explicit flags
    if show_lora:
        artifact_type = "lora_adapter"
    elif show_checkpoint:
        artifact_type = "training_checkpoint"
    elif show_gguf:
        artifact_type = "gguf_model"

    # GGUF-specific inspection
    if show_gguf and target.is_file() and target.suffix.lower() == ".gguf":
        _inspect_gguf_file(target, json_output=json_output, show_tensors=show_tensors)
        return

    # LoRA-specific inspection
    if show_lora and target.is_dir():
        _inspect_lora_dir(target, json_output=json_output)
        return

    # Checkpoint-specific inspection
    if show_checkpoint and target.is_dir():
        _inspect_checkpoint_dir(target, json_output=json_output)
        return

    if json_output:
        data: dict = {
            "artifact_type": artifact_type,
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
    print(f"Artifact type: {artifact_type}")

    if dir_info:
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
        if results:
            r = results[0]
            print(f"Detected type: {r.format.value}")
            print(f"File size: {_format_size(r.file_size)}")
            print(f"Details: {r.details}")

    print()

    if not dir_info or show_tensors:
        for r in results:
            rel = r.path.name if r.path.parent == target else str(r.path.relative_to(target))
            size_str = f" ({_format_size(r.file_size)})" if r.file_size else ""
            print(f"  {rel} [{r.format.value}]{size_str}")
            print(f"    {r.details}")

            if r.tensors and show_tensors:
                shown = min(10, len(r.tensors))
                for t in r.tensors[:shown]:
                    shape_str = "x".join(str(d) for d in t.shape)
                    print(f"      {t.name}: {t.dtype} [{shape_str}] ({_format_size(t.byte_size)})")
                if len(r.tensors) > shown:
                    print(f"      ... and {len(r.tensors) - shown} more tensors")

            if r.extra:
                for key, value in r.extra.items():
                    if key not in ("header_size",):
                        print(f"      {key}: {value}")


def _inspect_gguf_file(path: Path, json_output: bool = False, show_tensors: bool = False) -> None:
    """Display detailed GGUF file information."""
    from .formats.gguf import read_gguf_info

    try:
        info = read_gguf_info(path, parse_tensors=True)
    except (ValueError, OSError) as e:
        print(f"Error reading GGUF file: {e}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        data = {
            "artifact_type": "gguf_model",
            "path": str(path),
            "format": "gguf",
            "version": info.version,
            "endianness": info.endianness,
            "tensor_count": info.tensor_count,
            "metadata_kv_count": info.metadata_kv_count,
            "file_size": info.file_size,
            "quantization_summary": info.quantization_summary,
            "warnings": info.warnings,
        }
        if show_tensors and info.tensors:
            data["tensors"] = [
                {
                    "name": t.name,
                    "shape": t.shape,
                    "ggml_type": str(t.ggml_type),
                    "offset": t.offset,
                    "estimated_size": t.estimated_size,
                }
                for t in info.tensors
            ]
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    print("KMC GGUF Inspection")
    print()
    print("Detected type: GGUF model")
    print(f"Version: {info.version}")
    print(f"Endianness: {info.endianness}")
    print(f"Tensors: {info.tensor_count}")
    print(f"Metadata KV pairs: {info.metadata_kv_count}")
    print(f"File size: {_format_size(info.file_size)}")

    if info.warnings:
        print()
        print("Warnings:")
        for w in info.warnings:
            print(f"  {w}")

    if info.quantization_summary:
        print()
        print("Quantization summary:")
        for qtype, count in sorted(info.quantization_summary.items()):
            print(f"  {qtype}: {count}")
    else:
        print()
        print("Quantization summary: unknown")

    if show_tensors and info.tensors:
        print()
        shown = min(20, len(info.tensors))
        print(f"Tensors (showing {shown}/{len(info.tensors)}):")
        for t in info.tensors[:shown]:
            shape_str = "x".join(str(d) for d in t.shape)
            size_str = f" ({_format_size(t.estimated_size)})" if t.estimated_size else ""
            print(f"  {t.name}: {t.ggml_type} [{shape_str}]{size_str}")
        if len(info.tensors) > shown:
            print(f"  ... and {len(info.tensors) - shown} more tensors")


def _inspect_lora_dir(path: Path, json_output: bool = False) -> None:
    """Display LoRA adapter information."""
    from .workflows.lora import detect_lora_adapter

    adapter_info = detect_lora_adapter(path)

    if json_output:
        data = {
            "artifact_type": "lora_adapter",
            "path": str(path),
            "is_lora": adapter_info.is_lora,
            "has_adapter_model": adapter_info.has_adapter_model,
            "has_adapter_config": adapter_info.has_adapter_config,
            "has_readme": adapter_info.has_readme,
            "base_model_name_or_path": adapter_info.base_model_name_or_path,
            "peft_type": adapter_info.peft_type,
            "lora_rank": adapter_info.lora_rank,
            "target_modules": adapter_info.target_modules,
            "warnings": adapter_info.warnings,
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print("KMC LoRA Adapter Inspection")
    print()
    print(f"Path: {path}")
    print(f"Is LoRA: {'yes' if adapter_info.is_lora else 'no'}")
    print(f"Has adapter model: {'yes' if adapter_info.has_adapter_model else 'no'}")
    print(f"Has adapter config: {'yes' if adapter_info.has_adapter_config else 'no'}")
    print(f"PEFT type: {adapter_info.peft_type}")
    if adapter_info.lora_rank is not None:
        print(f"Rank: {adapter_info.lora_rank}")
    if adapter_info.target_modules:
        print(f"Target modules: {', '.join(adapter_info.target_modules)}")
    print(f"Base model: {adapter_info.base_model_name_or_path}")
    if adapter_info.warnings:
        print()
        print("Warnings:")
        for w in adapter_info.warnings:
            print(f"  {w}")


def _inspect_checkpoint_dir(path: Path, json_output: bool = False) -> None:
    """Display training checkpoint information."""
    from .workflows.checkpoint import detect_checkpoint

    ckpt_info = detect_checkpoint(path)

    if json_output:
        data = {
            "artifact_type": "training_checkpoint",
            "path": str(path),
            "is_checkpoint": ckpt_info.is_checkpoint,
            "step": ckpt_info.step,
            "has_trainer_state": ckpt_info.has_trainer_state,
            "has_optimizer_state": ckpt_info.has_optimizer_state,
            "has_scheduler_state": ckpt_info.has_scheduler_state,
            "has_rng_state": ckpt_info.has_rng_state,
            "has_safetensors_model": ckpt_info.has_safetensors_model,
            "has_pytorch_model": ckpt_info.has_pytorch_model,
            "detected_files": ckpt_info.detected_files,
            "warnings": ckpt_info.warnings,
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print("KMC Training Checkpoint Inspection")
    print()
    print(f"Path: {path}")
    print(f"Is checkpoint: {'yes' if ckpt_info.is_checkpoint else 'no'}")
    if ckpt_info.step is not None:
        print(f"Step: {ckpt_info.step}")
    print(f"Has trainer state: {'yes' if ckpt_info.has_trainer_state else 'no'}")
    print(f"Has optimizer state: {'yes' if ckpt_info.has_optimizer_state else 'no'}")
    print(f"Has scheduler state: {'yes' if ckpt_info.has_scheduler_state else 'no'}")
    print(f"Has RNG state: {'yes' if ckpt_info.has_rng_state else 'no'}")
    print(f"Has safetensors model: {'yes' if ckpt_info.has_safetensors_model else 'no'}")
    print(f"Has pytorch model: {'yes' if ckpt_info.has_pytorch_model else 'no'}")

    if ckpt_info.detected_files:
        print()
        print("Detected files:")
        for fname, category in sorted(ckpt_info.detected_files.items()):
            print(f"  {fname} ({category})")

    if ckpt_info.warnings:
        print()
        print("Warnings:")
        for w in ckpt_info.warnings:
            print(f"  {w}")


def _get_directory_model_info(directory: Path) -> dict:
    """Analyze a directory for model-level information."""
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

    shard_index = directory / "model.safetensors.index.json"
    if shard_index.is_file():
        is_sharded = True

    for f in directory.rglob("*.gguf"):
        if f.is_file():
            has_gguf = True

    lora_info = detect_lora_adapter(directory)
    is_lora = lora_info.get("is_lora", False)

    # Determine artifact type (for the new artifact_type field)
    if is_lora:
        detected_type_artifact = "lora_adapter"
        detected_type = "PEFT/LoRA adapter"
    elif has_safetensors and has_gguf:
        detected_type_artifact = "huggingface_model"
        detected_type = "Mixed format model"
    elif has_safetensors:
        detected_type_artifact = "huggingface_model"
        detected_type = "Hugging Face model folder"
    elif has_gguf:
        detected_type_artifact = "gguf_model"
        detected_type = "GGUF model"
    else:
        detected_type_artifact = "unknown"
        detected_type = "unknown"

    # Check for checkpoint patterns
    from .workflows.checkpoint import detect_checkpoint

    ckpt_info = detect_checkpoint(directory)
    if ckpt_info.is_checkpoint and not is_lora:
        detected_type_artifact = "training_checkpoint"
        detected_type = "Training checkpoint"

    return {
        "detected_type": detected_type,
        "detected_type_artifact": detected_type_artifact,
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

    # Partial-access benchmark
    partial_access = getattr(args, "partial_access", False)
    if partial_access:
        _run_partial_access_bench(args)
        return

    compare_zipnn = getattr(args, "compare_zipnn", False)
    compare_codecs = getattr(args, "compare_codecs", False)
    tensor_aware = getattr(args, "tensor_aware", False)
    codec = getattr(args, "codec", "auto")

    result = run_benchmark(
        source,
        output,
        synthetic=getattr(args, "synthetic", False),
        tensor_aware=tensor_aware,
        compare_zipnn=compare_zipnn,
        compare_codecs=compare_codecs,
        codec=codec,
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


def _run_partial_access_bench(args: argparse.Namespace) -> None:
    """Run a partial-access benchmark on a .kmc archive."""
    archive = Path(args.source)

    if not archive.exists():
        print(f"Error: archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    if not archive.suffix.lower() == ".kmc":
        print(f"Error: --partial-access requires a .kmc archive, got: {archive}", file=sys.stderr)
        sys.exit(1)

    import time as _time

    json_output = getattr(args, "json", False)
    only_pattern = getattr(args, "only", None)
    tensor_name = getattr(args, "partial_tensor", None)

    results: dict = {"archive": str(archive)}

    # Measure archive open + index build time
    t0 = _time.perf_counter()
    reader = KMCReader(archive)
    open_time = _time.perf_counter() - t0
    results["open_time_s"] = round(open_time, 6)

    files = reader.list_files()
    tensors = reader.list_tensors()
    results["total_files"] = len(files)
    results["total_tensors"] = len(tensors)
    results["total_blocks"] = reader.block_index.total_blocks

    # Measure reading a small file
    if files:
        # Find the smallest file
        smallest_file = None
        smallest_size = float("inf")
        for f_path in files:
            info = reader.get_file_info(f_path)
            if info and info.size < smallest_size:
                smallest_size = info.size
                smallest_file = f_path

        if smallest_file:
            t0 = _time.perf_counter()
            data = reader.read_file(smallest_file)
            read_file_time = _time.perf_counter() - t0
            results["read_small_file"] = {
                "path": smallest_file,
                "size": len(data),
                "time_s": round(read_file_time, 6),
            }

    # Measure reading a specific file by pattern
    if only_pattern:
        import fnmatch

        matched = [
            f
            for f in files
            if fnmatch.fnmatch(f, only_pattern) or fnmatch.fnmatch(f.split("/")[-1], only_pattern)
        ]
        if matched:
            t0 = _time.perf_counter()
            data = reader.read_file(matched[0])
            read_pattern_time = _time.perf_counter() - t0
            results["read_pattern_file"] = {
                "path": matched[0],
                "size": len(data),
                "time_s": round(read_pattern_time, 6),
            }

    # Measure reading a tensor
    if tensor_name and tensors:
        if reader.get_tensor_info(tensor_name):
            t0 = _time.perf_counter()
            tdata = reader.read_tensor(tensor_name)
            read_tensor_time = _time.perf_counter() - t0
            results["read_tensor"] = {
                "name": tensor_name,
                "size": len(tdata),
                "time_s": round(read_tensor_time, 6),
            }

    reader.close()

    if json_output:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("=== KMC Partial-Access Benchmark ===")
        print(f"Archive: {archive}")
        print(f"Open + index build time: {open_time:.6f}s")
        print(
            f"Files: {results['total_files']}, "
            f"Tensors: {results['total_tensors']}, "
            f"Blocks: {results['total_blocks']}"
        )
        if "read_small_file" in results:
            r = results["read_small_file"]
            print(f"Read small file ({r['path']}): {r['time_s']:.6f}s ({r['size']} bytes)")
        if "read_pattern_file" in results:
            r = results["read_pattern_file"]
            print(f"Read pattern file ({r['path']}): {r['time_s']:.6f}s ({r['size']} bytes)")
        if "read_tensor" in results:
            r = results["read_tensor"]
            print(f"Read tensor ({r['name']}): {r['time_s']:.6f}s ({r['size']} bytes)")


def _selective_unpack_files(
    reader: KMCReader,
    patterns: list[str],
    output_dir: Path,
    json_output: bool,
) -> None:
    """Extract files matching patterns from an archive."""
    import fnmatch

    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = reader.list_files()
    matched: list[str] = []
    bytes_written = 0

    for pattern in patterns:
        # Security: reject absolute paths and path traversal
        if pattern.startswith("/"):
            print(f"Error: absolute pattern not allowed: {pattern!r}", file=sys.stderr)
            sys.exit(1)
        if ".." in pattern.split("/"):
            print(f"Error: path traversal not allowed: {pattern!r}", file=sys.stderr)
            sys.exit(1)

        for f_path in all_files:
            basename = f_path.split("/")[-1]
            if fnmatch.fnmatch(f_path, pattern) or fnmatch.fnmatch(basename, pattern):
                if f_path not in matched:
                    matched.append(f_path)

    if not matched:
        print("Error: no files matched the given pattern(s)", file=sys.stderr)
        sys.exit(1)

    for f_path in matched:
        out_path = reader.extract_file(f_path, output_dir)
        data_len = out_path.stat().st_size
        bytes_written += data_len
        if not json_output:
            print(f"  Extracted: {f_path} ({_format_size(data_len)})")

    skipped = len(all_files) - len(matched)

    if json_output:
        print(
            json.dumps(
                {
                    "extracted": matched,
                    "skipped": skipped,
                    "bytes_written": bytes_written,
                },
                indent=2,
            )
        )
    else:
        print(
            f"Extracted {len(matched)} file(s), "
            f"skipped {skipped}, "
            f"wrote {_format_size(bytes_written)}"
        )


def _selective_unpack_tensor(
    reader: KMCReader,
    tensor_name: str,
    output_dir: Path,
    json_output: bool,
) -> None:
    """Extract a single tensor from an archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = reader.extract_tensor(tensor_name, output_dir)
    data_len = out_path.stat().st_size

    if json_output:
        print(
            json.dumps(
                {
                    "extracted_tensor": tensor_name,
                    "output_path": str(out_path),
                    "bytes_written": data_len,
                },
                indent=2,
            )
        )
    else:
        print(f"  Extracted tensor: {tensor_name} ({_format_size(data_len)}) -> {out_path}")


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
    p_pack.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "byteplane", "floatplane", "zstd", "zlib", "raw"],
        help="Compression codec (default: auto)",
    )
    p_pack.add_argument(
        "--gguf-aware",
        action="store_true",
        help="Experimental: use GGUF-aware compression mode",
    )
    p_pack.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 'auto' for cpu_count)",
    )
    p_pack.add_argument(
        "--progress",
        action="store_true",
        help="Show progress during operation",
    )
    p_pack.add_argument(
        "--dedup",
        action="store_true",
        help="Enable cross-file deduplication (experimental)",
    )
    p_pack.add_argument(
        "--delta-base",
        type=str,
        default=None,
        help="Path to base .kmc archive for delta compression (experimental)",
    )
    p_pack.set_defaults(func=cmd_pack)

    # pack-lora
    p_pack_lora = sub.add_parser(
        "pack-lora", help="Pack a LoRA adapter directory into a .kmc archive"
    )
    p_pack_lora.add_argument("source", help="LoRA adapter directory")
    p_pack_lora.add_argument("output", help="Output .kmc archive path")
    p_pack_lora.add_argument(
        "-b", "--block-size", type=int, default=None, help="Block size in bytes"
    )
    p_pack_lora.add_argument("-l", "--level", type=int, default=3, help="Compression level")
    p_pack_lora.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "byteplane", "floatplane", "zstd", "zlib", "raw"],
        help="Compression codec (default: auto)",
    )
    p_pack_lora.set_defaults(func=cmd_pack_lora)

    # pack-checkpoint
    p_pack_ckpt = sub.add_parser(
        "pack-checkpoint", help="Pack a training checkpoint directory into a .kmc archive"
    )
    p_pack_ckpt.add_argument("source", help="Training checkpoint directory")
    p_pack_ckpt.add_argument("output", help="Output .kmc archive path")
    p_pack_ckpt.add_argument(
        "-b", "--block-size", type=int, default=None, help="Block size in bytes"
    )
    p_pack_ckpt.add_argument("-l", "--level", type=int, default=3, help="Compression level")
    p_pack_ckpt.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "byteplane", "floatplane", "zstd", "zlib", "raw"],
        help="Compression codec (default: auto)",
    )
    p_pack_ckpt.set_defaults(func=cmd_pack_checkpoint)

    # unpack
    p_unpack = sub.add_parser("unpack", help="Unpack a .kmc archive")
    p_unpack.add_argument("archive", help=".kmc archive path")
    p_unpack.add_argument("output", help="Output directory")
    p_unpack.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    p_unpack.add_argument(
        "--progress",
        action="store_true",
        help="Show progress during operation",
    )
    p_unpack.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Extract only files matching pattern(s) (e.g., '*.json', 'config.json')",
    )
    p_unpack.add_argument(
        "--tensor",
        default=None,
        help="Extract a specific tensor by name",
    )
    p_unpack.add_argument(
        "--list",
        action="store_true",
        dest="list",
        help="List available files without extracting",
    )
    p_unpack.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    p_unpack.set_defaults(func=cmd_unpack)

    # verify
    p_verify = sub.add_parser("verify", help="Verify a .kmc archive integrity (full report)")
    p_verify.add_argument("archive", help=".kmc archive path")
    p_verify.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="verify_mode",
        help="Quick verify: check manifest and block hashes without decompressing",
    )
    p_verify.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="verify_mode",
        default="full",
        help="Full verify: decompress all blocks and verify file hashes (default)",
    )
    p_verify.set_defaults(func=cmd_verify, verify_mode="full")

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
    p_inspect.add_argument(
        "--compression",
        action="store_true",
        help="Show compression summary with codec usage",
    )
    p_inspect.add_argument(
        "--lora",
        action="store_true",
        help="Inspect as LoRA adapter",
    )
    p_inspect.add_argument(
        "--checkpoint",
        action="store_true",
        help="Inspect as training checkpoint",
    )
    p_inspect.add_argument(
        "--gguf",
        action="store_true",
        help="Inspect as GGUF model with tensor details",
    )
    p_inspect.add_argument("--dedup", action="store_true", help="Show deduplication info")
    p_inspect.add_argument(
        "--delta",
        action="store_true",
        dest="delta_info",
        help="Show delta compression info",
    )
    p_inspect.add_argument(
        "--runtime-hints",
        action="store_true",
        dest="runtime_hints",
        help="Show runtime integration hints",
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
        "--codec",
        default="auto",
        choices=["auto", "byteplane", "floatplane", "zstd", "zlib", "raw"],
        help="Compression codec (default: auto)",
    )
    p_bench.add_argument(
        "--compare-zipnn",
        action="store_true",
        help="Compare with ZipNN if available",
    )
    p_bench.add_argument(
        "--compare-codecs",
        action="store_true",
        help="Compare all available codecs",
    )
    p_bench.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    p_bench.add_argument(
        "--compare-jobs",
        default=None,
        help="Compare different job counts (e.g., '1,2,4,auto')",
    )
    p_bench.add_argument(
        "--progress",
        action="store_true",
        help="Show progress during operation",
    )
    p_bench.add_argument(
        "--partial-access",
        action="store_true",
        help=(
            "Benchmark partial access: measure time to open archive, "
            "build index, and read individual files/tensors"
        ),
    )
    p_bench.add_argument(
        "--only",
        default=None,
        help="With --partial-access: only read matching file pattern",
    )
    p_bench.add_argument(
        "--partial-tensor",
        default=None,
        help="With --partial-access: read a specific tensor by name",
    )
    p_bench.add_argument(
        "--dedup", action="store_true", help="Benchmark with deduplication enabled"
    )
    p_bench.set_defaults(func=cmd_bench)

    # list
    p_list = sub.add_parser("list", help="List the contents of a .kmc archive")
    p_list.add_argument("archive", help=".kmc archive path")
    p_list.add_argument(
        "--files",
        action="store_true",
        help="Show only files",
    )
    p_list.add_argument(
        "--tensors",
        action="store_true",
        help="Show only tensors",
    )
    p_list.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    p_list.set_defaults(func=cmd_list)

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
