"""zlib codec: standard DEFLATE compression (always available).

Uses Python's built-in zlib module for compression. This is the
baseline compression codec available on all platforms without
external dependencies.
"""

from __future__ import annotations

import zlib

from .base import CodecContext, CodecResult


class ZlibCodec:
    """zlib compression codec — always available baseline.

    Uses DEFLATE algorithm via Python's built-in zlib module.
    Compression level is configurable (default 6).
    """

    name: str = "zlib"

    def __init__(self, level: int = 6) -> None:
        self.level = level

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress data with zlib."""
        compressed = zlib.compress(data, level=self.level)
        return CodecResult(
            codec="zlib",
            payload=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            metadata={"level": self.level},
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress zlib-compressed data."""
        return zlib.decompress(payload)
