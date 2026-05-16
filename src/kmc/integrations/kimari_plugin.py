"""Kimari CLI plugin integration for KMC.

Provides a clean plugin interface that the Kimari CLI can use to register
KMC commands without circular dependencies. The Kimari project imports this
module and calls register_kimari_commands() to add compression, decompression,
verification, inspection, and benchmarking to its CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compress_model_command(
    source: str | Path,
    output: str | Path,
    tensor_aware: bool = True,
    gguf_aware: bool = False,
    codec: str = "auto",
    jobs: int = 1,
    level: int = 3,
    block_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Compress a model directory or file into a .kmc archive.

    This is the primary integration point for the Kimari CLI's
    ``kimari compress`` command.

    Args:
        source: Source directory or file.
        output: Output .kmc archive path.
        tensor_aware: Enable tensor-aware mode (default True for Kimari).
        gguf_aware: Enable GGUF-aware mode.
        codec: Compression codec.
        jobs: Number of parallel workers.
        level: Compression level.
        block_size: Block size in bytes (None = default).
        show_progress: Show progress output.

    Returns:
        Dict with status, sizes, ratio, and metadata.
    """
    from ..archive import DEFAULT_BLOCK_SIZE, pack

    source = Path(source).resolve()
    output = Path(output).resolve()

    if not source.exists():
        return {"status": "error", "message": f"Source not found: {source}"}

    import time

    start = time.time()
    pack(
        source,
        output,
        block_size=block_size or DEFAULT_BLOCK_SIZE,
        level=level,
        tensor_aware=tensor_aware,
        codec=codec,
        gguf_aware=gguf_aware,
        jobs=jobs,
    )
    elapsed = time.time() - start

    orig = (
        source.stat().st_size
        if source.is_file()
        else sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    )
    comp = output.stat().st_size
    ratio = comp / orig if orig > 0 else 0

    return {
        "status": "ok",
        "source": str(source),
        "output": str(output),
        "original_size": orig,
        "compressed_size": comp,
        "ratio": ratio,
        "elapsed_s": elapsed,
        "tensor_aware": tensor_aware,
        "gguf_aware": gguf_aware,
        "codec": codec,
        "jobs": jobs,
    }


def decompress_model_command(
    archive: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Decompress a .kmc archive to a directory.

    Integration point for ``kimari decompress``.

    Args:
        archive: Path to the .kmc archive.
        output_dir: Output directory.

    Returns:
        Dict with status and metadata.
    """
    from ..archive import unpack

    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()

    if not archive.exists():
        return {"status": "error", "message": f"Archive not found: {archive}"}

    import time

    start = time.time()
    unpack(archive, output_dir)
    elapsed = time.time() - start

    return {
        "status": "ok",
        "archive": str(archive),
        "output_dir": str(output_dir),
        "elapsed_s": elapsed,
    }


def verify_compressed_model_command(
    archive: str | Path,
    quick: bool = False,
) -> dict[str, Any]:
    """Verify a .kmc archive's integrity.

    Integration point for ``kimari verify-compress``.

    Args:
        archive: Path to the .kmc archive.
        quick: Use quick verification (no decompression).

    Returns:
        Dict with status, integrity, and any errors.
    """
    from ..archive import verify_full, verify_quick

    archive = Path(archive).resolve()

    if not archive.exists():
        return {"status": "error", "message": f"Archive not found: {archive}"}

    report = verify_quick(archive) if quick else verify_full(archive)

    return {
        "status": "ok" if report.integrity == "OK" else "failed",
        "archive": str(archive),
        "integrity": report.integrity,
        "errors": report.errors,
        "warnings": report.warnings,
        "total_files": report.total_files,
        "total_blocks": report.total_blocks,
        "quick": quick,
    }


def bench_compressed_model_command(
    source: str | Path,
    output: str | Path,
    codec: str = "auto",
    tensor_aware: bool = True,
    compare_codecs: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Benchmark compression performance.

    Integration point for ``kimari bench-compress``.

    Args:
        source: Source directory or file.
        output: Output .kmc archive path.
        codec: Compression codec.
        tensor_aware: Use tensor-aware mode.
        compare_codecs: Compare all available codecs.
        jobs: Number of parallel workers.

    Returns:
        Dict with benchmark results.
    """
    from ..benchmark import run_benchmark

    source = Path(source).resolve()
    output = Path(output).resolve()

    if not source.exists():
        return {"status": "error", "message": f"Source not found: {source}"}

    result = run_benchmark(
        source,
        output,
        tensor_aware=tensor_aware,
        compare_codecs=compare_codecs,
        codec=codec,
    )

    return {
        "status": "ok",
        "source": str(source),
        "jobs": jobs,
        "benchmark": result,
    }


def inspect_model_command(
    target: str | Path,
    json_output: bool = True,
    tensors: bool = False,
    lora: bool = False,
    checkpoint: bool = False,
    gguf: bool = False,
) -> dict[str, Any]:
    """Inspect a model directory, file, or .kmc archive.

    Integration point for ``kimari inspect-model``.

    Args:
        target: Path to inspect.
        json_output: Return structured data.
        tensors: Show tensor details.
        lora: Inspect as LoRA adapter.
        checkpoint: Inspect as training checkpoint.
        gguf: Inspect as GGUF model.

    Returns:
        Dict with inspection results.
    """
    from ..archive import inspect as archive_inspect

    target = Path(target).resolve()

    if not target.exists():
        return {"status": "error", "message": f"Target not found: {target}"}

    # If it's a .kmc archive, return manifest info
    if target.is_file() and target.suffix.lower() == ".kmc":
        manifest = archive_inspect(target)
        return {
            "status": "ok",
            "type": "kmc_archive",
            "path": str(target),
            "version": manifest.version,
            "artifact_type": manifest.artifact_type,
            "original_size": manifest.total_original_size,
            "compressed_size": manifest.total_compressed_size,
            "files": len(manifest.files),
            "parallelism": manifest.parallelism,
        }

    # For directories, use the inspector
    from ..inspector import inspect_directory, inspect_file

    if target.is_dir():
        results = inspect_directory(target)
        return {
            "status": "ok",
            "type": "model_directory",
            "path": str(target),
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

    # Single file
    result = inspect_file(target)
    return {
        "status": "ok",
        "type": "model_file",
        "path": str(target),
        "format": result.format.value,
        "details": result.details,
        "file_size": result.file_size,
    }


def register_kimari_commands(app_or_cli: Any) -> None:
    """Register KMC commands with a Kimari CLI application.

    This is the primary entry point for Kimari integration. The Kimari
    CLI calls this function during initialization to register all KMC
    commands as Kimari subcommands.

    Usage::

        from kmc.integrations.kimari_plugin import register_kimari_commands
        register_kimari_commands(kimari_app)

    Args:
        app_or_cli: The Kimari CLI application or click/argparse group
            to register commands with. Must have an add_command() method.
    """
    commands = {
        "compress": compress_model_command,
        "decompress": decompress_model_command,
        "verify-compress": verify_compressed_model_command,
        "bench-compress": bench_compressed_model_command,
        "inspect-model": inspect_model_command,
    }

    if hasattr(app_or_cli, "add_command"):
        for name, func in commands.items():
            try:
                app_or_cli.add_command(name, func)
            except (TypeError, AttributeError):
                pass


# Command mapping for documentation and discovery
KIMARI_PLUGIN_COMMAND_MAP = {
    "kimari compress": compress_model_command,
    "kimari decompress": decompress_model_command,
    "kimari verify-compress": verify_compressed_model_command,
    "kimari bench-compress": bench_compressed_model_command,
    "kimari inspect-model": inspect_model_command,
}
