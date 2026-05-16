"""Base codec interface: Protocol and data structures for all KMC codecs.

Defines the Codec protocol, CodecContext for tensor-aware hints,
and CodecResult for compression/decompression output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CodecContext:
    """Contextual hints for tensor-aware codec selection.

    Provides metadata about the data being compressed so codecs can
    make informed decisions about transformations (e.g., byte-plane
    separation for BF16/FP16 data).

    Attributes:
        file_path: Source file path, if known.
        tensor_name: Tensor name within the file, if known.
        dtype: Data type string (e.g., 'BF16', 'FP16', 'FP32', 'INT8').
        shape: Tensor shape as a list of integers.
        original_size: Original data size in bytes.
        block_index: Block index within the file.
    """

    file_path: str | None = None
    tensor_name: str | None = None
    dtype: str | None = None
    shape: list[int] | None = None
    original_size: int | None = None
    block_index: int | None = None


@dataclass
class CodecResult:
    """Result of a compression or decompression operation.

    Attributes:
        codec: Name of the codec that produced this result.
        payload: Compressed (or decompressed) data bytes.
        original_size: Size of the original data in bytes.
        compressed_size: Size of the compressed payload in bytes.
        metadata: Codec-specific metadata for reconstruction (e.g.,
            transform parameters, inner codec used, element_size).
    """

    codec: str
    payload: bytes
    original_size: int
    compressed_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Compression ratio (smaller is better; 1.0 = no compression)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


class Codec(Protocol):
    """Protocol for lossless compression codecs.

    Every codec must implement compress and decompress with guaranteed
    roundtrip exactness: decompress(compress(data)) == data for all inputs.
    """

    name: str

    def compress(self, data: bytes, *, context: CodecContext | None = None) -> CodecResult:
        """Compress data, optionally using context hints.

        Args:
            data: Input bytes to compress.
            context: Optional tensor-aware context hints.

        Returns:
            CodecResult with compressed payload and metadata.
        """
        ...

    def decompress(self, payload: bytes, *, context: CodecContext | None = None) -> bytes:
        """Decompress payload back to original data.

        Args:
            payload: Compressed bytes from a prior compress() call.
            context: Optional context (metadata dict may be needed).

        Returns:
            Original uncompressed bytes.
        """
        ...
