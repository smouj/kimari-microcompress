"""Codec layer: lossless compression codecs for KMC archives.

Provides a clean codec interface with tensor-aware extensions:
    - raw: passthrough (no compression)
    - zlib: standard zlib compression (always available)
    - zstd: Zstandard compression (optional, best ratio/speed)
    - byteplane: byte-plane separation for fixed-width numeric types
    - floatplane: sign/exponent/mantissa separation for FP16/BF16/FP32

All codecs are lossless. Roundtrip exactness is guaranteed.
If a codec does not improve size, it falls back to raw or zstd.
"""

from __future__ import annotations

from .base import Codec, CodecContext, CodecResult
from .byteplane import BytePlaneCodec
from .floatplane import FloatPlaneCodec
from .raw import RawCodec
from .registry import (
    get_codec,
    list_codecs,
    register_codec,
)
from .zlib_codec import ZlibCodec
from .zstd_codec import ZstdCodec

__all__ = [
    "Codec",
    "CodecContext",
    "CodecResult",
    "RawCodec",
    "ZlibCodec",
    "ZstdCodec",
    "BytePlaneCodec",
    "FloatPlaneCodec",
    "get_codec",
    "list_codecs",
    "register_codec",
]
