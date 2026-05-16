"""Compression codecs: zstd (preferred), zlib (fallback), raw (passthrough)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

try:
    import zstandard as zstd

    _HAS_ZSTD = True
except ImportError:  # pragma: no cover
    _HAS_ZSTD = False


class CodecId(str, Enum):
    """Supported compression codec identifiers."""

    ZSTD = "zstd"
    ZLIB = "zlib"
    RAW = "raw"


@dataclass(frozen=True)
class CodecResult:
    """Result of a compression/decompression operation."""

    data: bytes
    codec: CodecId
    original_size: int
    compressed_size: int

    @property
    def ratio(self) -> float:
        """Compression ratio (smaller is better; 1.0 = no compression)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


class Codec(Protocol):
    """Protocol for compression codecs."""

    def compress(self, data: bytes, level: int = 3) -> CodecResult: ...
    def decompress(self, data: bytes, original_size: int) -> CodecResult: ...


class ZstdCodec:
    """Zstandard compression codec — preferred for best ratio and speed."""

    def compress(self, data: bytes, level: int = 3) -> CodecResult:
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed")
        cctx = zstd.ZstdCompressor(level=level)
        compressed = cctx.compress(data)
        return CodecResult(
            data=compressed,
            codec=CodecId.ZSTD,
            original_size=len(data),
            compressed_size=len(compressed),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package not installed")
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(data, max_output_size=original_size)
        return CodecResult(
            data=decompressed,
            codec=CodecId.ZSTD,
            original_size=original_size,
            compressed_size=len(data),
        )


class ZlibCodec:
    """zlib compression codec — fallback when zstd is unavailable."""

    def compress(self, data: bytes, level: int = 6) -> CodecResult:
        compressed = zlib.compress(data, level=level)
        return CodecResult(
            data=compressed,
            codec=CodecId.ZLIB,
            original_size=len(data),
            compressed_size=len(compressed),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        decompressed = zlib.decompress(data)
        return CodecResult(
            data=decompressed,
            codec=CodecId.ZLIB,
            original_size=original_size,
            compressed_size=len(data),
        )


class RawCodec:
    """Passthrough codec — stores data uncompressed when compression doesn't help."""

    def compress(self, data: bytes, level: int = 0) -> CodecResult:
        return CodecResult(
            data=data,
            codec=CodecId.RAW,
            original_size=len(data),
            compressed_size=len(data),
        )

    def decompress(self, data: bytes, original_size: int) -> CodecResult:
        return CodecResult(
            data=data,
            codec=CodecId.RAW,
            original_size=original_size,
            compressed_size=len(data),
        )


def select_codec(data: bytes, block_size: int = 256 * 1024) -> Codec:
    """Select the best codec for the given data.

    Uses zstd if available, falls back to zlib. If the compressed output
    would not be smaller than the original, RawCodec is recommended instead.
    """
    if _HAS_ZSTD:
        return ZstdCodec()
    return ZlibCodec()


def compress_block(data: bytes, level: int = 3) -> CodecResult:
    """Compress a block using the best available codec.

    If compression doesn't reduce size, falls back to raw storage.
    """
    codec = select_codec(data)
    result = codec.compress(data, level=level)

    # If compression didn't help, store raw
    if result.compressed_size >= result.original_size:
        raw = RawCodec()
        return raw.compress(data)

    return result


def decompress_block(data: bytes, codec_id: CodecId, original_size: int) -> CodecResult:
    """Decompress a block given its codec identifier and original size."""
    codecs: dict[CodecId, Codec] = {
        CodecId.ZSTD: ZstdCodec(),
        CodecId.ZLIB: ZlibCodec(),
        CodecId.RAW: RawCodec(),
    }
    codec = codecs[codec_id]
    return codec.decompress(data, original_size)
