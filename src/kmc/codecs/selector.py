"""Automatic codec selector: choose the best lossless codec per block.

Selects codecs based on tensor metadata (dtype, shape) and tries
candidates in priority order. For each candidate, performs a local
roundtrip test and chooses the smallest result. Falls back safely
if any codec fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Codec, CodecContext, CodecResult
from .byteplane import BytePlaneCodec
from .floatplane import FloatPlaneCodec
from .raw import RawCodec
from .registry import is_codec_available
from .zlib_codec import ZlibCodec
from .zstd_codec import ZstdCodec, is_zstd_available

# Candidate codec chains per dtype category
_DTYPE_CANDIDATES: dict[str, list[str]] = {
    "BF16": ["floatplane", "byteplane", "zstd", "zlib", "raw"],
    "FP16": ["floatplane", "byteplane", "zstd", "zlib", "raw"],
    "F16": ["floatplane", "byteplane", "zstd", "zlib", "raw"],
    "FP32": ["floatplane", "byteplane", "zstd", "zlib", "raw"],
    "F32": ["floatplane", "byteplane", "zstd", "zlib", "raw"],
    "INT8": ["zstd", "zlib", "raw"],
    "UINT8": ["zstd", "zlib", "raw"],
    "INT16": ["zstd", "zlib", "raw"],
    "UINT16": ["zstd", "zlib", "raw"],
    "INT32": ["zstd", "zlib", "raw"],
    "UINT32": ["zstd", "zlib", "raw"],
    "FP64": ["zstd", "zlib", "raw"],
    "F64": ["zstd", "zlib", "raw"],
}

# Default candidates for unknown dtypes or GGUF
_DEFAULT_CANDIDATES = ["zstd", "zlib", "raw"]
_GGUF_CANDIDATES = ["zstd", "zlib", "raw"]


@dataclass
class SelectionResult:
    """Result of automatic codec selection for a single block.

    Attributes:
        result: The best CodecResult found.
        codec_name: Name of the selected codec.
        candidates_tried: Number of candidates that were attempted.
        roundtrip_verified: Whether the roundtrip was verified.
    """

    result: CodecResult
    codec_name: str
    candidates_tried: int
    roundtrip_verified: bool


def _normalize_dtype(dtype: str | None) -> str | None:
    """Normalize dtype string to a canonical form."""
    if dtype is None:
        return None
    upper = dtype.upper().strip()
    aliases = {
        "BFLOAT16": "BF16",
        "FLOAT16": "FP16",
        "FLOAT32": "FP32",
        "FLOAT64": "FP64",
    }
    return aliases.get(upper, upper)


def get_candidates(dtype: str | None, is_gguf: bool = False) -> list[str]:
    """Get ordered list of codec candidates for a given dtype.

    Args:
        dtype: Data type string (e.g., 'BF16', 'FP16', 'FP32').
        is_gguf: Whether the source is a GGUF file.

    Returns:
        List of codec names in priority order.
    """
    if is_gguf:
        return _GGUF_CANDIDATES

    normalized = _normalize_dtype(dtype)
    if normalized and normalized in _DTYPE_CANDIDATES:
        return _DTYPE_CANDIDATES[normalized]

    return _DEFAULT_CANDIDATES


def _instantiate_codec(name: str) -> Codec | None:
    """Try to instantiate a codec by name, returning None on failure."""
    if not is_codec_available(name):
        return None
    try:
        if name == "raw":
            return RawCodec()
        elif name == "zlib":
            return ZlibCodec()
        elif name == "zstd":
            return ZstdCodec() if is_zstd_available() else None
        elif name == "byteplane":
            return BytePlaneCodec()
        elif name == "floatplane":
            return FloatPlaneCodec()
        else:
            return None
    except Exception:
        return None


def select_codec(
    data: bytes,
    context: CodecContext | None = None,
    codec_override: str | None = None,
    verify_roundtrip: bool = True,
) -> SelectionResult:
    """Select the best codec for a block of data.

    If codec_override is specified, only that codec is tried.
    Otherwise, candidates are determined by dtype context and tried
    in priority order. The smallest compressed result wins.

    All candidates that successfully compress are roundtrip-tested
    (decompress(compress(data)) == data) unless verify_roundtrip=False.

    Args:
        data: Input bytes to compress.
        context: Tensor-aware context hints (dtype, shape, etc.).
        codec_override: Force a specific codec name (e.g., 'byteplane').
        verify_roundtrip: Whether to verify roundtrip exactness.

    Returns:
        SelectionResult with the best codec and its result.
    """
    # Determine candidates
    if codec_override:
        candidates = [codec_override]
    else:
        is_gguf = False
        if context and context.file_path and context.file_path.lower().endswith(".gguf"):
            is_gguf = True
        dtype = context.dtype if context else None
        candidates = get_candidates(dtype, is_gguf=is_gguf)

    best_result: CodecResult | None = None
    best_codec_name: str = "raw"
    candidates_tried = 0

    for codec_name in candidates:
        codec = _instantiate_codec(codec_name)
        if codec is None:
            continue

        try:
            result = codec.compress(data, context=context)
            candidates_tried += 1

            # Skip if compression didn't help and we have a better option
            if result.compressed_size >= result.original_size and best_result is not None:
                continue

            # Verify roundtrip
            if verify_roundtrip:
                try:
                    decomp_ctx = context
                    if decomp_ctx is None:
                        decomp_ctx = CodecContext(original_size=result.original_size)
                    # Attach codec metadata for decompressors that need it
                    decomp_ctx._codec_metadata = result.metadata  # type: ignore[attr-defined]

                    decompressed = codec.decompress(result.payload, context=decomp_ctx)
                    if decompressed != data:
                        continue  # Roundtrip failed, skip this codec
                except Exception:
                    continue  # Decompression error, skip

            # This codec is valid — check if it's the best so far
            if best_result is None or result.compressed_size < best_result.compressed_size:
                best_result = result
                best_codec_name = codec_name

        except Exception:
            continue

    # Fallback to raw if nothing worked
    if best_result is None:
        raw = RawCodec()
        best_result = raw.compress(data, context=context)
        best_codec_name = "raw"
        candidates_tried += 1

    return SelectionResult(
        result=best_result,
        codec_name=best_codec_name,
        candidates_tried=candidates_tried,
        roundtrip_verified=verify_roundtrip,
    )
