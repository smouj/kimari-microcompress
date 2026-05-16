"""AI model format inspector: detect safetensors, GGUF, LoRA, shards, and more.

Detects model formats by examining file magic bytes, structure, and naming
conventions. For safetensors, attempts to read real tensor metadata when
the format is available. For GGUF, reads header information. Degrades
gracefully when dependencies are unavailable.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelFormat(str, Enum):
    """Detected AI model format."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH_BIN = "pytorch_bin"
    PYTORCH_PT = "pytorch_pt"
    CHECKPOINT = "checkpoint"
    LORA_ADAPTER = "lora_adapter"
    TOKENIZER = "tokenizer"
    CONFIG = "config"
    MODEL_INDEX = "model_index"
    SHARD = "shard"
    UNKNOWN = "unknown"


@dataclass
class TensorMeta:
    """Metadata about a single tensor (from safetensors header)."""

    name: str
    dtype: str
    shape: list[int]
    byte_offset: int
    byte_size: int


@dataclass
class InspectionResult:
    """Result of inspecting a file for AI model format."""

    path: Path
    format: ModelFormat
    details: str
    file_size: int = 0
    tensors: list[TensorMeta] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian


def _read_magic(path: Path, n: int = 8) -> bytes:
    """Read the first n bytes of a file."""
    with open(path, "rb") as f:
        return f.read(n)


def _check_safetensors(path: Path) -> InspectionResult | None:
    """Check if a file is a safetensors file and read tensor metadata.

    safetensors format: first 8 bytes are the header length as a little-endian
    uint64, followed by a JSON header that starts with '{'.
    """
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]

            # Sanity check
            if header_len > 100_000_000 or header_len > file_size:
                return None

            header_data = f.read(header_len)
            if not header_data or header_data[0:1] != b"{":
                return None

            # Parse JSON header
            header = json.loads(header_data.decode("utf-8"))

            # Extract tensor metadata
            tensors: list[TensorMeta] = []
            total_params = 0
            dtypes: set[str] = set()

            for name, info in header.items():
                if name == "__metadata__":
                    continue

                dtype = info.get("dtype", "unknown")
                shape = info.get("shape", [])
                data_offsets = info.get("data_offsets", [0, 0])

                byte_offset = data_offsets[0] if len(data_offsets) >= 2 else 0
                byte_size = (data_offsets[1] - data_offsets[0]) if len(data_offsets) >= 2 else 0

                param_count = 1
                for dim in shape:
                    param_count *= dim

                total_params += param_count
                dtypes.add(dtype)

                tensors.append(
                    TensorMeta(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        byte_offset=byte_offset,
                        byte_size=byte_size,
                    )
                )

            largest = max(tensors, key=lambda t: t.byte_size) if tensors else None

            return InspectionResult(
                path=path,
                format=ModelFormat.SAFETENSORS,
                details=(
                    f"safetensors: {len(tensors)} tensors, "
                    f"{total_params:,} params, "
                    f"dtypes=[{', '.join(sorted(dtypes))}]"
                ),
                file_size=file_size,
                tensors=tensors,
                extra={
                    "total_params": total_params,
                    "dtypes": sorted(dtypes),
                    "largest_tensor": largest.name if largest else None,
                    "header_size": 8 + header_len,
                },
            )

    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


def _check_gguf(path: Path) -> InspectionResult | None:
    """Check if a file is a GGUF file and read header info."""
    try:
        file_size = path.stat().st_size
        magic_bytes = _read_magic(path, 4)
        if len(magic_bytes) < 4:
            return None
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != GGUF_MAGIC:
            return None

        # Read version and counts
        with open(path, "rb") as f:
            f.read(4)  # skip magic
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return InspectionResult(
                    path=path,
                    format=ModelFormat.GGUF,
                    details="GGUF (unknown version)",
                    file_size=file_size,
                )
            version = struct.unpack("<I", version_bytes)[0]

            tensor_count_bytes = f.read(8)
            tensor_count = (
                struct.unpack("<Q", tensor_count_bytes)[0] if len(tensor_count_bytes) >= 8 else 0
            )

            kv_count_bytes = f.read(8)
            kv_count = struct.unpack("<Q", kv_count_bytes)[0] if len(kv_count_bytes) >= 8 else 0

        return InspectionResult(
            path=path,
            format=ModelFormat.GGUF,
            details=(f"GGUF v{version}: {tensor_count} tensors, {kv_count} metadata keys"),
            file_size=file_size,
            extra={
                "version": version,
                "tensor_count": tensor_count,
                "kv_count": kv_count,
                "note": "Block-aware GGUF compression is future research",
            },
        )
    except (OSError, struct.error):
        pass
    return None


def _check_pytorch_bin(path: Path) -> InspectionResult | None:
    """Check if a file is a PyTorch .bin file (pickle-based)."""
    name = path.name.lower()
    if not name.endswith(".bin"):
        return None
    try:
        file_size = path.stat().st_size
        magic = _read_magic(path, 4)
        pickle_magics = [b"\x80\x02", b"\x80\x03", b"\x80\x04"]
        if any(magic[:2] == pm for pm in pickle_magics):
            return InspectionResult(
                path=path,
                format=ModelFormat.PYTORCH_BIN,
                details="PyTorch pickle protocol detected (insecure — prefer safetensors)",
                file_size=file_size,
            )
    except OSError:
        pass
    return None


def _check_pytorch_pt(path: Path) -> InspectionResult | None:
    """Check if a file is a PyTorch .pt file."""
    if path.suffix.lower() in (".pt", ".pth"):
        return InspectionResult(
            path=path,
            format=ModelFormat.PYTORCH_PT,
            details="PyTorch .pt file (pickle-based — prefer safetensors)",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def _check_checkpoint(path: Path) -> InspectionResult | None:
    """Check if a file is a checkpoint file (.ckpt)."""
    if path.suffix.lower() == ".ckpt":
        return InspectionResult(
            path=path,
            format=ModelFormat.CHECKPOINT,
            details="Checkpoint file",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def _check_lora_adapter(path: Path) -> InspectionResult | None:
    """Check if a file appears to be a LoRA adapter.

    LoRA adapters are typically safetensors files with 'lora' in the path
    or tensor names like 'lora_A.weight', 'lora_B.weight'.
    """
    name = path.name.lower()

    # Check naming convention
    is_lora_name = "lora" in name or "adapter" in name

    # If it's a safetensors, check for lora tensor names
    if path.suffix.lower() == ".safetensors":
        try:
            with open(path, "rb") as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    return None
                header_len = struct.unpack("<Q", header_len_bytes)[0]
                if header_len > 100_000_000:
                    return None
                header_data = f.read(header_len)
                header = json.loads(header_data.decode("utf-8"))

                tensor_names = [k for k in header if k != "__metadata__"]
                has_lora_tensors = any("lora" in tn.lower() for tn in tensor_names)

                if has_lora_tensors or is_lora_name:
                    lora_tensors = [tn for tn in tensor_names if "lora" in tn.lower()]
                    return InspectionResult(
                        path=path,
                        format=ModelFormat.LORA_ADAPTER,
                        details=(
                            f"LoRA adapter: {len(tensor_names)} tensors, "
                            f"{len(lora_tensors)} LoRA tensors"
                        ),
                        file_size=path.stat().st_size,
                        extra={"lora_tensors": lora_tensors[:20]},
                    )
        except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Also check for adapter_config.json companion
    if is_lora_name and path.suffix.lower() == ".safetensors":
        return InspectionResult(
            path=path,
            format=ModelFormat.LORA_ADAPTER,
            details="LoRA adapter (detected by name)",
            file_size=path.stat().st_size if path.is_file() else 0,
        )

    return None


def _check_tokenizer(path: Path) -> InspectionResult | None:
    """Check if a file is a tokenizer file."""
    name = path.name.lower()
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizers.json",
    }
    if name in tokenizer_names:
        return InspectionResult(
            path=path,
            format=ModelFormat.TOKENIZER,
            details=f"Tokenizer file: {name}",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def _check_config(path: Path) -> InspectionResult | None:
    """Check if a file is a model config file."""
    name = path.name.lower()
    config_names = {
        "config.json",
        "configuration.json",
        "model_config.json",
        "params.json",
    }
    if name in config_names:
        return InspectionResult(
            path=path,
            format=ModelFormat.CONFIG,
            details=f"Model config: {name}",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def _check_model_index(path: Path) -> InspectionResult | None:
    """Check if a file is a model index (sharded models)."""
    name = path.name.lower()
    if name in ("model_index.json", "models.json"):
        return InspectionResult(
            path=path,
            format=ModelFormat.MODEL_INDEX,
            details=f"Model index: {name}",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def _check_shard(path: Path) -> InspectionResult | None:
    """Check if a file is a model shard.

    Shards have names like:
    - model-00001-of-00002.safetensors
    - pytorch_model-00001-of-00002.bin
    """
    name = path.name.lower()
    # Match patterns like model-00001-of-00002.safetensors
    if "-of-" in name and (name.endswith(".safetensors") or name.endswith(".bin")):
        return InspectionResult(
            path=path,
            format=ModelFormat.SHARD,
            details=f"Model shard: {name}",
            file_size=path.stat().st_size if path.is_file() else 0,
        )
    return None


def inspect_file(path: Path) -> InspectionResult:
    """Inspect a file to detect its AI model format.

    Checks formats in priority order. For safetensors, reads real tensor
    metadata when possible. Degrades gracefully on errors.

    Args:
        path: Path to the file to inspect.

    Returns:
        InspectionResult with detected format and details.
    """
    path = Path(path)
    if not path.is_file():
        return InspectionResult(
            path=path,
            format=ModelFormat.UNKNOWN,
            details="Not a file",
        )

    # Check formats in priority order
    checks = [
        _check_lora_adapter,  # Before safetensors (more specific)
        _check_safetensors,
        _check_gguf,
        _check_shard,  # Before pytorch_bin (more specific)
        _check_pytorch_bin,
        _check_pytorch_pt,
        _check_checkpoint,
        _check_tokenizer,
        _check_config,
        _check_model_index,
    ]

    for check in checks:
        result = check(path)
        if result is not None:
            # Ensure file_size is set
            if result.file_size == 0:
                try:
                    result.file_size = path.stat().st_size
                except OSError:
                    pass
            return result

    return InspectionResult(
        path=path,
        format=ModelFormat.UNKNOWN,
        details="Unrecognized format",
        file_size=path.stat().st_size if path.is_file() else 0,
    )


def inspect_directory(directory: Path) -> list[InspectionResult]:
    """Inspect all files in a directory for AI model formats.

    Args:
        directory: Directory to scan.

    Returns:
        List of InspectionResult for each file found.
    """
    results: list[InspectionResult] = []
    directory = Path(directory)

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            results.append(inspect_file(path))

    return results
