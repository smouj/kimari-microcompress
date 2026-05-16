"""GGUF format parser: header reading, tensor metadata, and quantization summary.

Reads GGUF file metadata (magic, version, endianness, tensor count,
metadata KV count, tensor names, shapes, types, offsets, sizes, and
quantization summaries) without loading the full file into memory.

For GGUF v2/v3 files, this parser reads the tensor info section that
follows the header, extracting per-tensor metadata. If parsing fails
partially, the parser degrades gracefully with warnings.

GGUF format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_MAGIC_LE = 0x46475547  # "GGUF" read as little-endian uint32
GGUF_MAGIC_BE = 0x47554647  # "GGUF" read as big-endian uint32 (indicates big-endian file)

# GGML quantization type mapping (common types)
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "F64",
    28: "IQ1_M",
    29: "BF16",
    30: "Q4_0_4_4",
    31: "Q4_0_4_8",
    32: "Q4_0_8_8",
}

# Block sizes for quantized types (bytes per block, block_size elements)
GGML_BLOCK_SIZES: dict[int, int] = {
    0: 4,  # F32: 4 bytes per element
    1: 2,  # F16: 2 bytes per element
    2: 18,  # Q4_0: 18 bytes per block of 32
    3: 20,  # Q4_1: 20 bytes per block of 32
    6: 22,  # Q5_0: 22 bytes per block of 32
    7: 24,  # Q5_1: 24 bytes per block of 32
    8: 34,  # Q8_0: 34 bytes per block of 32
    9: 36,  # Q8_1: 36 bytes per block of 32
    10: 84,  # Q2_K
    11: 110,  # Q3_K
    12: 144,  # Q4_K
    13: 176,  # Q5_K
    14: 210,  # Q6_K
    15: 292,  # Q8_K
    29: 2,  # BF16: 2 bytes per element
}

# Elements per block for quantized types
GGML_ELEMENTS_PER_BLOCK: dict[int, int] = {
    0: 1,  # F32
    1: 1,  # F16
    2: 32,  # Q4_0
    3: 32,  # Q4_1
    6: 32,  # Q5_0
    7: 32,  # Q5_1
    8: 32,  # Q8_0
    9: 32,  # Q8_1
    10: 256,  # Q2_K
    11: 256,  # Q3_K
    12: 256,  # Q4_K
    13: 256,  # Q5_K
    14: 256,  # Q6_K
    15: 256,  # Q8_K
    29: 1,  # BF16
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GGUFTensorInfo:
    """Metadata about a single tensor in a GGUF file.

    Attributes:
        name: Tensor name (e.g., 'token_embd.weight').
        shape: Tensor dimensions (e.g., [4096, 32000]).
        ggml_type: GGML type as string name if known, else int ID.
        offset: Byte offset within the tensor data region (if available).
        estimated_size: Estimated byte size of the tensor data (if available).
    """

    name: str
    shape: list[int] = field(default_factory=list)
    ggml_type: str | int = "unknown"
    offset: int | None = None
    estimated_size: int | None = None


@dataclass
class GGUFInfo:
    """Parsed GGUF file information including tensor metadata.

    Attributes:
        available: Whether GGUF parsing was successful.
        magic: The raw magic bytes as a string (e.g., "GGUF").
        version: GGUF format version (1, 2, or 3).
        endianness: Either "little" or "big", determined from magic byte order.
        tensor_count: Number of tensors in the file (from header).
        metadata_kv_count: Number of metadata key-value pairs (from header).
        file_size: Total file size in bytes.
        header_size: Size of the header portion that was read.
        tensors: List of parsed tensor metadata entries.
        quantization_summary: Dict mapping quantization type name to count.
        warnings: List of warnings encountered during parsing.
        tensor_metadata_implemented: Always True now (tensor metadata parsing available).
    """

    available: bool = True
    magic: str = "GGUF"
    version: int = 0
    endianness: str = "little"
    tensor_count: int = 0
    metadata_kv_count: int = 0
    file_size: int = 0
    header_size: int = 0
    tensors: list[GGUFTensorInfo] = field(default_factory=list)
    quantization_summary: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tensor_metadata_implemented: bool = True


# ---------------------------------------------------------------------------
# GGUF string reading helper
# ---------------------------------------------------------------------------


def _read_gguf_string(f, fmt_prefix: str) -> str | None:
    """Read a GGUF string (length-prefixed) from the current file position.

    GGUF string format:
        - 8 bytes: string length (uint64)
        - N bytes: string data (UTF-8)

    Returns None if reading fails.
    """
    try:
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            return None
        str_len = struct.unpack(f"{fmt_prefix}Q", len_bytes)[0]
        # Sanity check
        if str_len > 1_000_000:
            return None
        str_data = f.read(str_len)
        if len(str_data) < str_len:
            return None
        return str_data.decode("utf-8", errors="replace")
    except (struct.error, OSError):
        return None


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def read_gguf_info(path: Path, parse_tensors: bool = True) -> GGUFInfo:
    """Read GGUF file information including tensor metadata.

    Opens the file in binary mode, reads the header and optionally
    the tensor info section, and returns a GGUFInfo structure.
    Does NOT load the entire file into memory.

    Args:
        path: Path to the GGUF file.
        parse_tensors: If True, attempt to parse per-tensor metadata
            (names, shapes, types, offsets). If False, only parse header.

    Returns:
        GGUFInfo with parsed information. Partial results are returned
        with warnings if some sections cannot be parsed.

    Raises:
        ValueError: If the file is not a valid GGUF file.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = path.stat().st_size
    warnings: list[str] = []

    # GGUF header layout (minimum 24 bytes for v2/v3):
    #   4 bytes: magic ("GGUF")
    #   4 bytes: version (uint32)
    #   8 bytes: tensor_count (uint64)
    #   8 bytes: metadata_kv_count (uint64)
    MIN_HEADER_SIZE = 24

    with open(path, "rb") as f:
        # Read magic
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("File too small for GGUF header")

        # Determine endianness from magic
        magic_le = struct.unpack("<I", magic_bytes)[0]
        magic_be = struct.unpack(">I", magic_bytes)[0]

        if magic_le == GGUF_MAGIC_LE:
            endianness = "little"
            fmt_prefix = "<"
        elif magic_be == GGUF_MAGIC_LE:
            endianness = "big"
            fmt_prefix = ">"
        else:
            raise ValueError(
                f"Invalid GGUF magic: 0x{magic_le:08X} (expected 0x{GGUF_MAGIC_LE:08X})"
            )

        # Read version
        version_bytes = f.read(4)
        if len(version_bytes) < 4:
            raise ValueError("Truncated GGUF version")
        version = struct.unpack(f"{fmt_prefix}I", version_bytes)[0]

        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Read tensor_count and metadata_kv_count
        tensor_count = 0
        metadata_kv_count = 0

        if version >= 2:
            tc_bytes = f.read(8)
            if len(tc_bytes) >= 8:
                tensor_count = struct.unpack(f"{fmt_prefix}Q", tc_bytes)[0]

            kv_bytes = f.read(8)
            if len(kv_bytes) >= 8:
                metadata_kv_count = struct.unpack(f"{fmt_prefix}Q", kv_bytes)[0]
        elif version == 1:
            # v1 uses uint32 for counts
            tc_bytes = f.read(4)
            if len(tc_bytes) >= 4:
                tensor_count = struct.unpack(f"{fmt_prefix}I", tc_bytes)[0]

            kv_bytes = f.read(4)
            if len(kv_bytes) >= 4:
                metadata_kv_count = struct.unpack(f"{fmt_prefix}I", kv_bytes)[0]

        # Parse tensor metadata if requested
        tensors: list[GGUFTensorInfo] = []
        quantization_summary: dict[str, int] = {}

        if parse_tensors and version >= 2 and tensor_count > 0:
            # After the header, we need to skip the metadata KV pairs
            # to reach the tensor info section.
            # The layout is:
            #   [header: 24 bytes]
            #   [metadata KV pairs: variable]
            #   [tensor info: variable]
            #   [padding to alignment]
            #   [tensor data]

            # Skip metadata KV pairs by reading and discarding each one
            # This is necessary because the tensor info section comes after
            try:
                for i in range(min(metadata_kv_count, 10000)):
                    # Read key string
                    key = _read_gguf_string(f, fmt_prefix)
                    if key is None:
                        warnings.append(
                            f"Failed to read metadata key {i}, skipping remaining KV pairs"
                        )
                        break

                    # Read value type
                    vtype_bytes = f.read(4)
                    if len(vtype_bytes) < 4:
                        warnings.append("Truncated metadata value type")
                        break
                    vtype = struct.unpack(f"{fmt_prefix}I", vtype_bytes)[0]

                    # Skip value based on type
                    if not _skip_gguf_value(f, vtype, fmt_prefix):
                        warnings.append(
                            f"Failed to skip metadata value for key '{key}' "
                            f"(type={vtype}), skipping remaining KV pairs"
                        )
                        break
            except (struct.error, OSError) as e:
                warnings.append(f"Error reading metadata KV pairs: {e}")

            # Now we should be at the tensor info section
            try:
                for i in range(min(tensor_count, 100000)):
                    tensor_name = _read_gguf_string(f, fmt_prefix)
                    if tensor_name is None:
                        warnings.append(f"Failed to read tensor name {i}, stopping tensor parsing")
                        break

                    # Read n_dimensions
                    ndim_bytes = f.read(4)
                    if len(ndim_bytes) < 4:
                        warnings.append(f"Truncated tensor dimensions for '{tensor_name}'")
                        break
                    n_dimensions = struct.unpack(f"{fmt_prefix}I", ndim_bytes)[0]

                    # Read dimensions
                    shape: list[int] = []
                    for _ in range(min(n_dimensions, 8)):
                        dim_bytes = f.read(8)
                        if len(dim_bytes) < 8:
                            warnings.append(f"Truncated shape for tensor '{tensor_name}'")
                            break
                        shape.append(struct.unpack(f"{fmt_prefix}Q", dim_bytes)[0])

                    # Read type
                    type_bytes = f.read(4)
                    if len(type_bytes) < 4:
                        warnings.append(f"Truncated type for tensor '{tensor_name}'")
                        break
                    type_id = struct.unpack(f"{fmt_prefix}I", type_bytes)[0]

                    # Read offset
                    offset_bytes = f.read(8)
                    if len(offset_bytes) < 8:
                        warnings.append(f"Truncated offset for tensor '{tensor_name}'")
                        break
                    offset = struct.unpack(f"{fmt_prefix}Q", offset_bytes)[0]

                    # Resolve type name
                    ggml_type: str | int = GGML_TYPE_NAMES.get(type_id, type_id)

                    # Estimate tensor size
                    estimated_size = _estimate_tensor_size(type_id, shape)

                    tensor_info = GGUFTensorInfo(
                        name=tensor_name,
                        shape=shape,
                        ggml_type=ggml_type,
                        offset=offset,
                        estimated_size=estimated_size,
                    )
                    tensors.append(tensor_info)

                    # Update quantization summary
                    type_name = str(ggml_type)
                    quantization_summary[type_name] = quantization_summary.get(type_name, 0) + 1

            except (struct.error, OSError) as e:
                warnings.append(f"Error reading tensor info: {e}")

            if len(tensors) < tensor_count:
                warnings.append(
                    f"Partially parsed tensor metadata: got {len(tensors)}/{tensor_count} tensors"
                )

        elif parse_tensors and version == 1:
            warnings.append("GGUF v1 tensor metadata parsing is not supported")

        elif not parse_tensors:
            pass  # Skip tensor parsing as requested

    header_size = MIN_HEADER_SIZE if version >= 2 else 16

    return GGUFInfo(
        available=True,
        magic="GGUF",
        version=version,
        endianness=endianness,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        file_size=file_size,
        header_size=header_size,
        tensors=tensors,
        quantization_summary=quantization_summary,
        warnings=warnings,
        tensor_metadata_implemented=True,
    )


def _skip_gguf_value(f, vtype: int, fmt_prefix: str) -> bool:
    """Skip a GGUF metadata value based on its type.

    Returns True if successfully skipped, False otherwise.
    """
    # GGUF value types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    try:
        if vtype == GGUF_TYPE_UINT8 or vtype == GGUF_TYPE_INT8:
            f.read(1)
        elif vtype == GGUF_TYPE_UINT16 or vtype == GGUF_TYPE_INT16:
            f.read(2)
        elif vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL):
            f.read(4)
        elif vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
            f.read(8)
        elif vtype == GGUF_TYPE_STRING:
            s = _read_gguf_string(f, fmt_prefix)
            if s is None:
                return False
        elif vtype == GGUF_TYPE_ARRAY:
            # Array: type (uint32) + count (uint64) + values
            arr_type_bytes = f.read(4)
            if len(arr_type_bytes) < 4:
                return False
            arr_type = struct.unpack(f"{fmt_prefix}I", arr_type_bytes)[0]
            arr_count_bytes = f.read(8)
            if len(arr_count_bytes) < 8:
                return False
            arr_count = struct.unpack(f"{fmt_prefix}Q", arr_count_bytes)[0]

            # Sanity check
            if arr_count > 1_000_000:
                return False

            for _ in range(arr_count):
                if not _skip_gguf_value(f, arr_type, fmt_prefix):
                    return False
        else:
            return False
        return True
    except (struct.error, OSError):
        return False


def _estimate_tensor_size(type_id: int, shape: list[int]) -> int | None:
    """Estimate the byte size of a tensor based on its type and shape.

    Returns None if the type is unknown and size cannot be estimated.
    """
    if not shape:
        return None

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if type_id == 0:  # F32
        return total_elements * 4
    elif type_id == 1:  # F16
        return total_elements * 2
    elif type_id == 29:  # BF16
        return total_elements * 2
    elif type_id in (24,):  # I8
        return total_elements * 1
    elif type_id in (25,):  # I16
        return total_elements * 2
    elif type_id in (26,):  # I32
        return total_elements * 4
    elif type_id in (27,):  # F64
        return total_elements * 8
    elif type_id in GGML_ELEMENTS_PER_BLOCK and type_id in GGML_BLOCK_SIZES:
        block_size = GGML_ELEMENTS_PER_BLOCK[type_id]
        block_bytes = GGML_BLOCK_SIZES[type_id]
        num_blocks = (total_elements + block_size - 1) // block_size
        return num_blocks * block_bytes

    return None


def is_gguf_file(path: Path) -> bool:
    """Check if a file is a valid GGUF file.

    This is a lightweight check that only reads the magic bytes.

    Args:
        path: Path to the file to check.

    Returns:
        True if the file appears to be a valid GGUF file.
    """
    try:
        with open(path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return False
            magic_le = struct.unpack("<I", magic_bytes)[0]
            magic_be = struct.unpack(">I", magic_bytes)[0]
            return magic_le == GGUF_MAGIC_LE or magic_be == GGUF_MAGIC_LE
    except (OSError, struct.error):
        return False


def is_quantized_ggml_type(type_name: str | int) -> bool:
    """Check if a GGML type is a quantized (non-float) type.

    Quantized types like Q4_K_M, Q5_0, Q8_0, etc. represent compressed
    weights that should not have floatplane/byteplane applied to them.

    Args:
        type_name: GGML type as string name or int ID.

    Returns:
        True if the type is quantized (not F32, F16, BF16).
    """
    float_types = {"F32", "F16", "BF16", "F64"}
    if isinstance(type_name, str):
        return type_name not in float_types
    # Integer type IDs
    return type_name not in (0, 1, 29, 27)
