"""Codec registry: discover, register, and instantiate codecs by name.

Provides a central registry for all available codecs, allowing
codecs to be looked up by name and instantiated with appropriate
configuration.
"""

from __future__ import annotations

from .base import Codec
from .byteplane import BytePlaneCodec
from .floatplane import FloatPlaneCodec
from .raw import RawCodec
from .zlib_codec import ZlibCodec
from .zstd_codec import ZstdCodec, is_zstd_available

# Global codec registry: name -> codec class
_REGISTRY: dict[str, type[Codec]] = {
    "raw": RawCodec,
    "zlib": ZlibCodec,
    "zstd": ZstdCodec,
    "byteplane": BytePlaneCodec,
    "floatplane": FloatPlaneCodec,
}


def register_codec(name: str, codec_cls: type[Codec]) -> None:
    """Register a custom codec by name.

    Args:
        name: Codec name (must be unique).
        codec_cls: Codec class implementing the Codec protocol.

    Raises:
        ValueError: If a codec with this name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Codec '{name}' is already registered")
    _REGISTRY[name] = codec_cls


def get_codec(name: str, **kwargs: object) -> Codec:
    """Get a codec instance by name.

    Args:
        name: Codec name (e.g., 'zstd', 'zlib', 'raw', 'byteplane', 'floatplane').
        **kwargs: Additional arguments passed to the codec constructor.

    Returns:
        Codec instance.

    Raises:
        ValueError: If the codec name is not registered.
        RuntimeError: If the codec's dependencies are not installed.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown codec: {name!r}. Available: {list_codecs()}")

    # Special handling for zstd availability
    if name == "zstd" and not is_zstd_available():
        raise RuntimeError("zstandard package not installed — pip install zstandard")

    cls = _REGISTRY[name]
    codec = cls(**kwargs)
    return codec


def list_codecs() -> list[str]:
    """List all registered codec names.

    Returns names of all codecs, regardless of whether their
    dependencies are installed. Use is_codec_available() to check.
    """
    return sorted(_REGISTRY.keys())


def is_codec_available(name: str) -> bool:
    """Check if a codec is available (registered AND dependencies met).

    Args:
        name: Codec name.

    Returns:
        True if the codec can be instantiated without errors.
    """
    if name not in _REGISTRY:
        return False
    if name == "zstd":
        return is_zstd_available()
    return True


def available_codecs() -> list[str]:
    """List only codecs whose dependencies are installed."""
    return [name for name in sorted(_REGISTRY.keys()) if is_codec_available(name)]
