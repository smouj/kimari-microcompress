"""Kimari CLI integration: maps kimari commands to KMC operations.

This module provides a clean adapter layer that maps Kimari CLI commands
to the underlying KMC operations:

    kimari compress           -> kmc pack [--tensor-aware] [--codec]
    kimari compress-lora      -> kmc pack-lora
    kimari compress-checkpoint -> kmc pack-checkpoint
    kimari decompress         -> kmc unpack
    kimari verify-compress    -> kmc verify
    kimari inspect-model      -> kmc inspect [--lora|--checkpoint|--gguf|--tensors]
    kimari bench-compress     -> kmc bench [--compare-zipnn] [--compare-codecs]

Usage from Kimari CLI:

    from kmc.integrations.kimari import (
        kimari_compress,
        kimari_compress_lora,
        kimari_compress_checkpoint,
        kimari_decompress,
        kimari_verify_compress,
        kimari_bench_compress,
        kimari_inspect_model,
        kimari_pack_lora,
        kimari_pack_checkpoint,
    )

This module does NOT modify Kimari itself. It provides the integration
surface that Kimari can call when the KMC package is available.
"""

from __future__ import annotations

from pathlib import Path

from ..archive import DEFAULT_BLOCK_SIZE, pack, unpack, verify_full
from ..benchmark import BenchmarkResult, run_benchmark
from ..workflows.checkpoint import (
    build_checkpoint_manifest_metadata,
    detect_checkpoint,
)
from ..workflows.lora import (
    build_lora_manifest_metadata,
    detect_lora_adapter,
)


def kimari_compress(
    source: str | Path,
    output: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    tensor_aware: bool = True,
    codec: str = "auto",
    gguf_aware: bool = False,
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
        gguf_aware: Use GGUF-aware compression mode (experimental).

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
        gguf_aware=gguf_aware,
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
        "gguf_aware": gguf_aware,
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


def kimari_pack_lora(
    input_path: str | Path,
    output_path: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    codec: str = "auto",
) -> dict:
    """Pack a LoRA adapter directory (kimari compress-lora).

    Args:
        input_path: LoRA adapter directory.
        output_path: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        codec: Compression codec.

    Returns:
        Dict with status, metadata, and sizes.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    adapter_info = detect_lora_adapter(input_path)
    if not adapter_info.is_lora:
        return {
            "status": "error",
            "message": f"Not a LoRA adapter directory: {input_path}",
        }

    # Pack with tensor-aware mode for safetensors
    pack(
        input_path,
        output_path,
        block_size=block_size,
        level=level,
        tensor_aware=True,
        codec=codec,
        artifact_type="lora_adapter",
        artifact_metadata=build_lora_manifest_metadata(adapter_info),
    )

    orig_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
    comp_size = output_path.stat().st_size
    ratio = comp_size / orig_size if orig_size > 0 else 0

    return {
        "status": "ok",
        "source": str(input_path),
        "output": str(output_path),
        "original_size": orig_size,
        "compressed_size": comp_size,
        "ratio": ratio,
        "artifact_type": "lora_adapter",
        "peft_type": adapter_info.peft_type,
        "lora_rank": adapter_info.lora_rank,
        "base_model": adapter_info.base_model_name_or_path,
        "target_modules": adapter_info.target_modules,
    }


def kimari_pack_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    level: int = 3,
    codec: str = "auto",
) -> dict:
    """Pack a training checkpoint directory (kimari compress-checkpoint).

    Args:
        input_path: Checkpoint directory.
        output_path: Output .kmc archive path.
        block_size: Block size in bytes.
        level: Compression level.
        codec: Compression codec.

    Returns:
        Dict with status, metadata, and sizes.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    ckpt_info = detect_checkpoint(input_path)
    if not ckpt_info.is_checkpoint:
        return {
            "status": "error",
            "message": f"Not a training checkpoint directory: {input_path}",
        }

    # Pack with tensor-aware mode if safetensors is present
    tensor_aware = ckpt_info.has_safetensors_model
    pack(
        input_path,
        output_path,
        block_size=block_size,
        level=level,
        tensor_aware=tensor_aware,
        codec=codec,
        artifact_type="training_checkpoint",
        artifact_metadata=build_checkpoint_manifest_metadata(ckpt_info),
    )

    orig_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
    comp_size = output_path.stat().st_size
    ratio = comp_size / orig_size if orig_size > 0 else 0

    return {
        "status": "ok",
        "source": str(input_path),
        "output": str(output_path),
        "original_size": orig_size,
        "compressed_size": comp_size,
        "ratio": ratio,
        "artifact_type": "training_checkpoint",
        "step": ckpt_info.step,
        "has_optimizer_state": ckpt_info.has_optimizer_state,
        "has_trainer_state": ckpt_info.has_trainer_state,
        "has_safetensors_model": ckpt_info.has_safetensors_model,
    }


def kimari_inspect_model(
    input_path: str | Path,
    json_output: bool = False,
) -> dict:
    """Inspect a model directory or file (kimari inspect-model).

    Args:
        input_path: Model directory or file to inspect.
        json_output: Whether to return JSON-serializable output.

    Returns:
        Dict with inspection results.
    """
    input_path = Path(input_path)

    from ..inspector import inspect_directory, inspect_file

    if input_path.is_dir():
        results = inspect_directory(input_path)
        # Detect artifact type
        adapter_info = detect_lora_adapter(input_path)
        ckpt_info = detect_checkpoint(input_path)

        artifact_type = "unknown"
        if adapter_info.is_lora:
            artifact_type = "lora_adapter"
        elif ckpt_info.is_checkpoint:
            artifact_type = "training_checkpoint"
        else:
            # Check for GGUF or HF model
            has_gguf = any(r.format.value == "gguf" for r in results)
            has_safetensors = any(
                r.format.value in ("safetensors", "lora_adapter") for r in results
            )
            if has_gguf:
                artifact_type = "gguf_model"
            elif has_safetensors:
                artifact_type = "huggingface_model"

        return {
            "status": "ok",
            "path": str(input_path),
            "artifact_type": artifact_type,
            "files": [
                {
                    "path": str(r.path),
                    "format": r.format.value,
                    "details": r.details,
                    "file_size": r.file_size,
                }
                for r in results
            ],
        }
    else:
        r = inspect_file(input_path)
        return {
            "status": "ok",
            "path": str(input_path),
            "artifact_type": r.format.value,
            "format": r.format.value,
            "details": r.details,
            "file_size": r.file_size,
        }


# Command mapping for documentation
KIMARI_COMMAND_MAP = {
    "kimari compress": (
        "kmc pack [--tensor-aware] [--codec auto|byteplane|floatplane|zstd|zlib|raw] [--gguf-aware]"
    ),
    "kimari compress-lora": "kmc pack-lora",
    "kimari compress-checkpoint": "kmc pack-checkpoint",
    "kimari decompress": "kmc unpack",
    "kimari verify-compress": "kmc verify",
    "kimari inspect-model": "kmc inspect [--lora|--checkpoint|--gguf|--tensors]",
    "kimari bench-compress": "kmc bench [--compare-zipnn] [--compare-codecs]",
}
