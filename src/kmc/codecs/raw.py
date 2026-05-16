"""Raw codec: passthrough with no compression.

Stores data uncompressed when compression doesn't reduce size.
Always available as the ultimate fallback.
"""

from __future__ import annotations

from .base import CodecContext, CodecResult


class RawCodec:
    """Passthrough codec — stores data uncompressed.

    Used when no compression codec reduces the data size, or as
    the mandatory fallback for any codec chain.
    """

    name: str = "raw"

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Store data uncompressed."""
        return CodecResult(
            codec="raw",
            payload=data,
            original_size=len(data),
            compressed_size=len(data),
            metadata={},
        )

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Return payload as-is (no decompression needed)."""
        return payload
