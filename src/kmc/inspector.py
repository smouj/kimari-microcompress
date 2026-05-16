"""AI model format inspector: detect safetensors, GGUF, .bin, .pt, .ckpt."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelFormat(str, Enum):
    """Detected AI model format."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH_BIN = "pytorch_bin"
    PYTORCH_PT = "pytorch_pt"
    CHECKPOINT = "checkpoint"
    UNKNOWN = "unknown"


@dataclass
class InspectionResult:
    """Result of inspecting a file for AI model format."""

    path: Path
    format: ModelFormat
    details: str


# Magic bytes for known formats
SAFETENSORS_MAGIC = 0x6E69616D736F6E65  # 8 bytes: "safetense" part
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian

# GGUF version constants
GGUF_V1 = 1
GGUF_V2 = 2
GGUF_V3 = 3


def _read_magic(path: Path, n: int = 8) -> bytes:
    """Read the first n bytes of a file."""
    with open(path, "rb") as f:
        return f.read(n)


def _check_safetensors(path: Path) -> InspectionResult | None:
    """Check if a file is a safetensors file.

    safetensors format: first 8 bytes are the header length as a little-endian
    uint64, followed by a JSON header that starts with '{'.
    """
    try:
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]

            # Sanity check: header shouldn't be absurdly large
            if header_len > 100_000_000:  # 100 MB max header
                return None

            header_data = f.read(min(header_len, 1024))  # Read start of header
            if header_data and header_data[0:1] == b"{":
                try:
                    # Try to parse beginning of JSON
                    json.loads(header_data.decode("utf-8") + "...}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                return InspectionResult(
                    path=path,
                    format=ModelFormat.SAFETENSORS,
                    details=f"safetensors header length: {header_len} bytes",
                )
    except (OSError, struct.error):
        pass
    return None


def _check_gguf(path: Path) -> InspectionResult | None:
    """Check if a file is a GGUF file.

    GGUF magic: 0x46475547 ("GGUF" as uint32 LE) at offset 0.
    """
    try:
        magic_bytes = _read_magic(path, 4)
        if len(magic_bytes) < 4:
            return None
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic == GGUF_MAGIC:
            # Read version
            with open(path, "rb") as f:
                f.read(4)  # skip magic
                version_bytes = f.read(4)
                if len(version_bytes) >= 4:
                    version = struct.unpack("<I", version_bytes)[0]
                    return InspectionResult(
                        path=path,
                        format=ModelFormat.GGUF,
                        details=f"GGUF version {version}",
                    )
                return InspectionResult(
                    path=path,
                    format=ModelFormat.GGUF,
                    details="GGUF (unknown version)",
                )
    except (OSError, struct.error):
        pass
    return None


def _check_pytorch_bin(path: Path) -> InspectionResult | None:
    """Check if a file is a PyTorch .bin file (pickle-based)."""
    name = path.name.lower()
    if name.endswith(".bin"):
        # PyTorch .bin files are typically pickle-based
        try:
            magic = _read_magic(path, 4)
            if magic[:2] == b"\x80\x02" or magic[:2] == b"\x80\x03" or magic[:2] == b"\x80\x04":
                return InspectionResult(
                    path=path,
                    format=ModelFormat.PYTORCH_BIN,
                    details="PyTorch pickle protocol detected",
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
            details="PyTorch .pt file",
        )
    return None


def _check_checkpoint(path: Path) -> InspectionResult | None:
    """Check if a file is a checkpoint file (.ckpt)."""
    if path.suffix.lower() == ".ckpt":
        return InspectionResult(
            path=path,
            format=ModelFormat.CHECKPOINT,
            details="Checkpoint file",
        )
    return None


def inspect_file(path: Path) -> InspectionResult:
    """Inspect a file to detect its AI model format.

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
        _check_safetensors,
        _check_gguf,
        _check_pytorch_bin,
        _check_pytorch_pt,
        _check_checkpoint,
    ]

    for check in checks:
        result = check(path)
        if result is not None:
            return result

    return InspectionResult(
        path=path,
        format=ModelFormat.UNKNOWN,
        details="Unrecognized format",
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
