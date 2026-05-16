"""Experimental loaders for partial tensor access from .kmc archives.

These loaders are experimental and may change without notice. They
provide tensor-byte and native tensor loading capabilities for archives
that include safetensors metadata.

KMC does not perform compressed inference. Tensors are fully decompressed
before being returned.
"""

from __future__ import annotations

from .safetensors_loader import load_tensor, load_tensor_bytes

__all__ = [
    "load_tensor",
    "load_tensor_bytes",
]
