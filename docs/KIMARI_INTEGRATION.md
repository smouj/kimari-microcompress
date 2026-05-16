# Kimari Integration

## Overview

Kimari MicroCompress is designed as a foundational component of the Kimari ecosystem. This document outlines the planned integration path between KMC and the broader Kimari platform.

## Integration Architecture

```
┌─────────────────────┐
│  Kimari Platform     │
│                      │
│  ┌────────────────┐  │
│  │ kimari compress│◄─┼── KMC CLI / Python API
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ kimari store   │◄─┼── .kmc archive storage
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ kimari verify  │◄─┼── SHA-256 integrity checks
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ kimari serve   │◄─┼── Block-level serving (future)
│  └────────────────┘  │
└─────────────────────┘
```

## Phase 1: `kimari compress` Command

The first integration point is a `kimari compress` command that wraps KMC's functionality:

```bash
kimari compress ./my-model --output ./my-model.kmc
kimari decompress ./my-model.kmc --output ./restored/
kimari verify ./my-model.kmc
```

This provides a Kimari-branded interface while delegating to KMC's core implementation.

### Implementation

```python
# kimari/compress.py (conceptual)
from kmc.archive import pack, unpack, verify as kmc_verify

def compress(source, output, **kwargs):
    """Compress a model using KMC."""
    pack(source, output, **kwargs)

def decompress(archive, output, **kwargs):
    """Decompress a KMC archive."""
    unpack(archive, output, **kwargs)

def verify(archive, **kwargs):
    """Verify a KMC archive's integrity."""
    return kmc_verify(archive)
```

## Phase 2: KimariDB Storage Backend

KMC archives can be stored in KimariDB with metadata indexing:

- Archive hash (SHA-256 of the .kmc file).
- Source model identifier (Hugging Face model ID, custom ID).
- Compression ratio and codec information.
- Creation timestamp and tool version.

This enables:
- Content-addressed storage (deduplication of identical archives).
- Fast lookup of compressed versions of known models.
- Verification that stored archives haven't been tampered with.

## Phase 3: Download-Cache-Verify Workflow

Integrate with Hugging Face Hub for a seamless download-compress-verify workflow:

```bash
kimari download huggingface/gpt2 --compress --verify
```

This would:
1. Download model files from Hugging Face.
2. Compress them into a .kmc archive.
3. Verify the archive integrity.
4. Store the archive in the local KimariDB cache.
5. Register the archive metadata for future lookups.

## Phase 4: Block-Level Serving

The most ambitious integration is block-level serving of model data:

1. KMC archives are stored in KimariDB with full block metadata.
2. When a model is loaded for inference, only the required blocks are fetched and decompressed.
3. This enables:
   - Loading specific layers on demand.
   - Streaming model loading for large models.
   - Memory-efficient inference on resource-constrained devices.

### API Concept

```python
# Block-level loading (conceptual)
from kimari.serve import BlockServer

server = BlockServer("huggingface/llama-7b.kmc")

# Load only the embedding layer and first 4 transformer layers
embedding = server.load_tensor("model.embed_tokens.weight")
layers_0_3 = server.load_range(start=0, count=4)
```

## Compatibility Notes

- KMC archives are self-contained and don't require the Kimari platform to use.
- The `kmc` CLI operates independently of any Kimari services.
- Kimari integration adds value on top of KMC's core functionality.
- Archives created with `kmc pack` can be used with future `kimari compress` features.

## Testing Integration

Integration tests should cover:
- Roundtrip via `kimari compress` → `kimari decompress`.
- Verification via `kimari verify` matches `kmc verify`.
- Metadata consistency between KMC and KimariDB.
- Backward compatibility of .kmc archives across KMC versions.
