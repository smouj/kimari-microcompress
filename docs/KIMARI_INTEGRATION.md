# Kimari Integration

## Overview

Kimari MicroCompress is designed as a foundational component of the Kimari ecosystem. This document describes the integration path between KMC and the Kimari CLI, including the adapter layer already implemented.

## Command Mapping

The integration maps Kimari CLI commands to KMC operations:

| Kimari Command | KMC Equivalent | Description |
|----------------|---------------|-------------|
| `kimari compress` | `kmc pack [--tensor-aware]` | Compress a model into a .kmc archive |
| `kimari decompress` | `kmc unpack` | Decompress a .kmc archive |
| `kimari verify-compress` | `kmc verify` | Verify archive integrity |
| `kimari bench-compress` | `kmc bench [--compare-zipnn]` | Benchmark compression performance |

## Integration Layer

The adapter module is implemented at `src/kmc/integrations/kimari.py` and provides:

```python
from kmc.integrations.kimari import (
    kimari_compress,        # -> kmc pack [--tensor-aware]
    kimari_decompress,      # -> kmc unpack
    kimari_verify_compress, # -> kmc verify
    kimari_bench_compress,  # -> kmc bench [--compare-zipnn]
    KIMARI_COMMAND_MAP,     # Command mapping dict
)
```

### Usage Example

```python
from kmc.integrations.kimari import kimari_compress, kimari_verify_compress

# Compress with tensor-aware mode (default for Kimari)
result = kimari_compress("./my-model", "./my-model.kmc", tensor_aware=True)
# result = {
#     "status": "ok",
#     "original_size": 1000000,
#     "compressed_size": 600000,
#     "ratio": 0.6,
#     "tensor_aware": True,
# }

# Verify
verify_result = kimari_verify_compress("./my-model.kmc")
# verify_result = {"status": "ok", "integrity": "OK", "errors": [], "total_files": 5}
```

### CLI Adapter Example

An example CLI adapter is provided at `examples/kimari_cli_adapter.py`:

```bash
python examples/kimari_cli_adapter.py compress ./model ./model.kmc
python examples/kimari_cli_adapter.py decompress ./model.kmc ./restored
python examples/kimari_cli_adapter.py verify-compress ./model.kmc
python examples/kimari_cli_adapter.py bench-compress ./model ./model.kmc --compare-zipnn
```

## Integration Architecture

```
┌─────────────────────────────────┐
│  Kimari CLI                      │
│                                  │
│  kimari compress ./model         │
│       ↓                          │
│  ┌─────────────────────────────┐ │
│  │ kmc.integrations.kimari    │ │  Adapter layer
│  │   kimari_compress()        │ │
│  │   (tensor_aware=True)      │ │
│  └─────────────────────────────┘ │
│       ↓                          │
│  ┌─────────────────────────────┐ │
│  │ kmc.archive                │ │  Core KMC
│  │   pack(tensor_aware=True)  │ │
│  └─────────────────────────────┘ │
│       ↓                          │
│  ┌─────────────────────────────┐ │
│  │ kmc.formats.safetensors    │ │  Format support
│  │   read_safetensors_info()  │ │
│  └─────────────────────────────┘ │
└─────────────────────────────────┘
```

## Design Principles

1. **KMC is independent**: The `kmc` CLI operates standalone. Kimari integration is optional.
2. **Adapter pattern**: The integration layer translates Kimari conventions to KMC calls without modifying KMC internals.
3. **Return values**: Integration functions return structured dicts or dataclasses, not CLI output.
4. **No circular dependencies**: KMC does not depend on Kimari. Kimari depends on KMC.
5. **Tensor-aware by default**: The Kimari adapter enables `tensor_aware=True` by default, since Kimari users are working with AI models and benefit from tensor-level metadata.

## Phase 1: Adapter Layer (Current — v0.3)

- [x] `kimari_compress()` adapter function with `tensor_aware` parameter
- [x] `kimari_decompress()` adapter function
- [x] `kimari_verify_compress()` adapter function
- [x] `kimari_bench_compress()` adapter function with `compare_zipnn` parameter
- [x] Command mapping documentation
- [x] CLI adapter example (`examples/kimari_cli_adapter.py`)

## Phase 2: Kimari CLI Integration (v0.6)

- [ ] Add `kimari compress` subcommand to Kimari CLI
- [ ] Add `kimari decompress` subcommand
- [ ] Add `kimari verify-compress` subcommand
- [ ] Add `kimari bench-compress` subcommand
- [ ] Shared configuration (block size, compression level)
- [ ] Progress reporting integration

## Phase 3: KimariDB Storage Backend

KMC archives can be stored in KimariDB with metadata indexing:

- Archive hash (SHA-256 of the .kmc file)
- Source model identifier (Hugging Face model ID, custom ID)
- Compression ratio and codec information
- Tensor metadata summary
- Creation timestamp and tool version

This enables:
- Content-addressed storage (deduplication of identical archives)
- Fast lookup of compressed versions of known models
- Verification that stored archives haven't been tampered with
- Tensor-level metadata queries (find archives containing BF16 tensors)

## Phase 4: Download-Cache-Verify Workflow

Integrate with Hugging Face Hub for a seamless workflow:

```bash
kimari download huggingface/gpt2 --compress --verify --tensor-aware
```

This would:
1. Download model files from Hugging Face
2. Compress them into a .kmc archive with tensor-aware mode
3. Verify the archive integrity
4. Store the archive in the local KimariDB cache
5. Register the archive metadata for future lookups

## Phase 5: Block-Level Serving

The most ambitious integration is block-level serving:

1. KMC archives are stored in KimariDB with full block metadata
2. When a model is loaded for inference, only the required blocks are fetched and decompressed
3. This enables loading specific layers on demand, streaming model loading, and memory-efficient inference

### API Concept

```python
from kmc.integrations.kimari import BlockServer

server = BlockServer("huggingface/llama-7b.kmc")

# Load only the embedding layer and first 4 transformer layers
embedding = server.load_tensor("model.embed_tokens.weight")
layers_0_3 = server.load_range(start=0, count=4)
```

## Testing Integration

Integration tests should cover:
- Roundtrip via `kimari_compress()` -> `kimari_decompress()`
- Verification via `kimari_verify_compress()` matches `kmc verify`
- Tensor-aware mode produces manifest with tensor entries
- ZipNN comparison works when zipnn is installed
- ZipNN gracefully degrades when zipnn is not installed
- Metadata consistency between KMC and KimariDB
- Backward compatibility of .kmc archives across KMC versions
- Error propagation from KMC to Kimari CLI
