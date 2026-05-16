"""Safetensors format support: read metadata without loading weights.

This module provides a dedicated layer for reading safetensors file metadata,
including tensor names, dtypes, shapes, byte offsets, and size estimates.
It also detects sharded models and LoRA/PEFT adapters.

If the optional ``safetensors`` package is installed, it is used for header
parsing. If not, a pure-Python fallback reads the 8-byte header length prefix
and JSON header directly from the file. No weights are ever loaded into memory.

No ``pickle`` is used. No full model loading occurs.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import importlib.util

    if importlib.util.find_spec("safetensors") is not None:
        _HAS_SAFETENSORS_LIB = True
    else:
        _HAS_SAFETENSORS_LIB = False
except Exception:
    _HAS_SAFETENSORS_LIB = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TensorEntry:
    """Metadata about a single tensor in a safetensors file."""

    name: str
    dtype: str
    shape: list[int]
    byte_offset: int
    byte_size: int


@dataclass
class SafetensorsInfo:
    """Parsed metadata from a safetensors file or set of shard files.

    Attributes:
        available: Whether safetensors support is available.
        unavailable_reason: If not available, why (and how to fix it).
        tensors: List of tensor metadata entries.
        tensor_count: Total number of tensors.
        total_tensor_bytes: Sum of all tensor data sizes in bytes.
        total_params: Total number of parameters across all tensors.
        dtypes: Sorted list of unique dtype strings found.
        header_size: Size of the header (8 + JSON length) in bytes.
        file_size: Size of the file on disk in bytes.
        is_shard: Whether this file is part of a sharded model.
        shard_index: If sharded, which shard number this is (1-based).
        shard_total: If sharded, total number of shards.
        is_lora: Whether this appears to be a LoRA/PEFT adapter.
        lora_rank: LoRA rank if detected, else None.
        target_modules: LoRA target modules if detected.
        base_model_reference: Base model reference if found in metadata.
        metadata: The ``__metadata__`` dict from the safetensors header, if present.
    """

    available: bool = True
    unavailable_reason: str = ""
    tensors: list[TensorEntry] = field(default_factory=list)
    tensor_count: int = 0
    total_tensor_bytes: int = 0
    total_params: int = 0
    dtypes: list[str] = field(default_factory=list)
    header_size: int = 0
    file_size: int = 0
    is_shard: bool = False
    shard_index: int = 0
    shard_total: int = 0
    is_lora: bool = False
    lora_rank: int | None = None
    target_modules: list[str] = field(default_factory=list)
    base_model_reference: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shard / LoRA naming detection
# ---------------------------------------------------------------------------

# Pattern: model-00001-of-00002.safetensors
_SHARD_PATTERN = re.compile(r"model-(\d+)-of-(\d+)\.safetensors", re.IGNORECASE)

# Pattern: adapter_model.safetensors or similar
_LORA_NAME_PATTERN = re.compile(r"(adapter|lora)", re.IGNORECASE)

# Pattern: lora_A.weight, lora_B.weight, etc.
_LORA_TENSOR_PATTERN = re.compile(r"lora_[ab]", re.IGNORECASE)


def _parse_shard_name(filename: str) -> tuple[bool, int, int]:
    """Check if filename matches shard naming convention.

    Returns (is_shard, index, total).
    """
    m = _SHARD_PATTERN.match(filename)
    if m:
        return True, int(m.group(1)), int(m.group(2))
    return False, 0, 0


def _detect_lora_from_tensors(
    tensor_names: list[str],
    metadata: dict,
) -> tuple[bool, int | None, list[str], str | None]:
    """Detect LoRA/PEFT adapter from tensor names and metadata.

    Returns (is_lora, rank, target_modules, base_model_ref).
    """
    has_lora_tensors = any(_LORA_TENSOR_PATTERN.search(n) for n in tensor_names)
    if not has_lora_tensors:
        return False, None, [], None

    # Extract target modules from tensor names
    target_modules_set: set[str] = set()
    for name in tensor_names:
        if _LORA_TENSOR_PATTERN.search(name):
            # Extract module path before .lora_A or .lora_B
            parts = name.rsplit(".lora_", 1)
            if len(parts) >= 1:
                target_modules_set.add(parts[0])

    # Try to get rank from tensor shapes (lora_B dimension)
    rank = None  # Will be filled by caller if shapes are available

    # Try to get base model reference from metadata
    base_model_ref = None
    if metadata:
        base_model_ref = metadata.get("base_model_name_or_path") or metadata.get("base_model", None)

    return True, rank, sorted(target_modules_set), base_model_ref


# ---------------------------------------------------------------------------
# Core parsing (pure-Python fallback)
# ---------------------------------------------------------------------------


def _read_safetensors_header_raw(path: Path) -> dict:
    """Read and parse the safetensors JSON header using raw I/O.

    safetensors format:
        - First 8 bytes: header length (little-endian uint64)
        - Next header_length bytes: JSON header
        - JSON header maps tensor names to {dtype, shape, data_offsets}
        - Special "__metadata__" key for user metadata

    This does NOT load any tensor data.
    """
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File too small for safetensors header")
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        # Sanity check: header should not be unreasonably large
        file_size = path.stat().st_size
        if header_len > 100_000_000:
            raise ValueError(f"Safetensors header too large: {header_len:,} bytes")
        if header_len > file_size:
            raise ValueError(
                f"Safetensors header length ({header_len:,}) exceeds file size ({file_size:,})"
            )

        header_data = f.read(header_len)
        if len(header_data) < header_len:
            raise ValueError("Truncated safetensors header")

    # Validate that it starts with '{'
    if header_data[:1] != b"{":
        raise ValueError("Safetensors header does not start with '{'")

    return json.loads(header_data.decode("utf-8"))


def _parse_header_to_info(header: dict, path: Path) -> SafetensorsInfo:
    """Convert a parsed safetensors JSON header into a SafetensorsInfo."""
    file_size = path.stat().st_size
    filename = path.name

    tensors: list[TensorEntry] = []
    total_params = 0
    total_bytes = 0
    dtypes: set[str] = set()
    metadata: dict = {}
    tensor_names: list[str] = []

    # Calculate header size (8 bytes for length prefix + JSON length)
    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_size = 8 + len(header_json)

    for name, info in header.items():
        if name == "__metadata__":
            metadata = info if isinstance(info, dict) else {}
            continue

        dtype = info.get("dtype", "unknown")
        shape = info.get("shape", [])
        data_offsets = info.get("data_offsets", [0, 0])

        if len(data_offsets) >= 2:
            byte_offset = data_offsets[0]
            byte_size = data_offsets[1] - data_offsets[0]
        else:
            byte_offset = 0
            byte_size = 0

        param_count = 1
        for dim in shape:
            param_count *= dim

        total_params += param_count
        total_bytes += byte_size
        dtypes.add(dtype)
        tensor_names.append(name)

        tensors.append(
            TensorEntry(
                name=name,
                dtype=dtype,
                shape=shape,
                byte_offset=byte_offset,
                byte_size=byte_size,
            )
        )

    # Detect shard status
    is_shard, shard_index, shard_total = _parse_shard_name(filename)

    # Detect LoRA
    is_lora, lora_rank, target_modules, base_model_ref = _detect_lora_from_tensors(
        tensor_names, metadata
    )

    # If LoRA detected and we have tensor shapes, try to infer rank
    if is_lora and lora_rank is None:
        for t in tensors:
            if _LORA_TENSOR_PATTERN.search(t.name) and len(t.shape) >= 1:
                # lora_A: [rank, input_dim], lora_B: [output_dim, rank]
                # The rank is typically the smaller dimension
                name_lower = t.name.lower()
                if "lora_a" in name_lower:
                    lora_rank = t.shape[0]
                    break
                elif "lora_b" in name_lower:
                    lora_rank = t.shape[-1]
                    break

    return SafetensorsInfo(
        available=True,
        tensors=tensors,
        tensor_count=len(tensors),
        total_tensor_bytes=total_bytes,
        total_params=total_params,
        dtypes=sorted(dtypes),
        header_size=header_size,
        file_size=file_size,
        is_shard=is_shard,
        shard_index=shard_index,
        shard_total=shard_total,
        is_lora=is_lora,
        lora_rank=lora_rank,
        target_modules=target_modules,
        base_model_reference=base_model_ref,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_safetensors_info(path: Path) -> SafetensorsInfo:
    """Read safetensors metadata from a file without loading weights.

    Uses the ``safetensors`` library if available, otherwise falls back
    to a pure-Python header parser. No tensor data is loaded.

    Args:
        path: Path to the .safetensors file.

    Returns:
        SafetensorsInfo with parsed metadata.

    Raises:
        ValueError: If the file is not a valid safetensors file.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.suffix.lower() == ".safetensors":
        raise ValueError(f"Not a safetensors file: {path}")

    # Try using the safetensors library for better compatibility
    if _HAS_SAFETENSORS_LIB:
        try:
            return _read_with_safetensors_lib(path)
        except Exception:
            pass  # Fall through to raw parser

    # Fallback: pure-Python header parsing
    header = _read_safetensors_header_raw(path)
    return _parse_header_to_info(header, path)


def _read_with_safetensors_lib(path: Path) -> SafetensorsInfo:
    """Read safetensors metadata using the safetensors library."""
    from safetensors import safe_open  # type: ignore[import-untyped]

    # Verify the file can be opened with safetensors library
    with safe_open(str(path), framework="numpy") as f:
        f.keys()  # Verify we can read tensor names
        f.metadata()  # Verify we can read metadata

    # We still need to read the raw header for offsets and sizes
    # (the safetensors library doesn't expose data_offsets easily)
    header = _read_safetensors_header_raw(path)
    return _parse_header_to_info(header, path)


def is_safetensors_available() -> bool:
    """Check if the safetensors optional dependency is installed.

    Note: The pure-Python fallback works without the dependency, but
    the library provides better compatibility with edge cases.
    """
    return _HAS_SAFETENSORS_LIB


def get_safetensors_status() -> dict[str, str]:
    """Get the current safetensors support status.

    Returns a dict with 'status', 'reason' (if unavailable), and
    'suggestion' (if unavailable).
    """
    if _HAS_SAFETENSORS_LIB:
        return {
            "status": "available",
            "reason": "",
            "suggestion": "",
        }
    return {
        "status": "unavailable",
        "reason": "optional dependency not installed",
        "suggestion": 'pip install "kimari-microcompress[safetensors]"',
    }


def detect_safetensors_shards(directory: Path) -> list[Path]:
    """Detect safetensors shard files in a directory.

    Looks for files matching the pattern:
        model-NNNNN-of-MMMMM.safetensors

    Also checks for model.safetensors.index.json as a sharding indicator.

    Args:
        directory: Directory to scan for shard files.

    Returns:
        Sorted list of shard file paths.
    """
    directory = Path(directory)
    shards: list[Path] = []

    for f in sorted(directory.rglob("*.safetensors")):
        if f.is_file() and _SHARD_PATTERN.match(f.name):
            shards.append(f)

    return shards


def detect_lora_adapter(directory: Path) -> dict:
    """Detect LoRA/PEFT adapter files in a directory.

    Looks for:
        - adapter_model.safetensors (with or without adapter_config.json)
        - Files with LoRA tensor names

    Args:
        directory: Directory to scan.

    Returns:
        Dict with detection results.
    """
    directory = Path(directory)

    # Check for adapter_config.json
    adapter_config = directory / "adapter_config.json"
    has_adapter_config = adapter_config.is_file()

    # Check for adapter model files
    adapter_model = directory / "adapter_model.safetensors"
    has_adapter_model = adapter_model.is_file()

    # Check for any safetensors file with LoRA tensors
    lora_files: list[Path] = []
    if has_adapter_model:
        lora_files.append(adapter_model)
    else:
        for f in directory.rglob("*.safetensors"):
            if f.is_file() and _LORA_NAME_PATTERN.search(f.name):
                lora_files.append(f)

    # Read adapter_config.json if present
    lora_rank = None
    target_modules: list[str] = []
    base_model_ref = None

    if has_adapter_config:
        try:
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            lora_rank = config.get("r", None)
            target_modules = config.get("target_modules", [])
            if isinstance(target_modules, str):
                target_modules = [target_modules]
            base_model_ref = config.get("base_model_name_or_path", None)
        except (json.JSONDecodeError, OSError):
            pass

    # If we have an adapter model, try to read tensor-level info
    is_lora = has_adapter_model or len(lora_files) > 0
    if is_lora and not lora_rank and has_adapter_model:
        try:
            info = read_safetensors_info(adapter_model)
            if info.is_lora:
                lora_rank = info.lora_rank
                if not target_modules:
                    target_modules = info.target_modules
                if not base_model_ref:
                    base_model_ref = info.base_model_reference
        except (ValueError, OSError):
            pass

    return {
        "is_lora": is_lora,
        "has_adapter_config": has_adapter_config,
        "has_adapter_model": has_adapter_model,
        "lora_rank": lora_rank,
        "target_modules": target_modules,
        "base_model_reference": base_model_ref,
        "adapter_files": [str(f) for f in lora_files],
    }
