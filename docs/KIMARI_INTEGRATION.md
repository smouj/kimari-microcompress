# Kimari Integration

## Overview

Kimari MicroCompress is designed as a foundational component of the Kimari ecosystem. This document describes the integration path between KMC and the Kimari CLI, including the adapter layer implemented in v0.5.

## Command Mapping

The integration maps Kimari CLI commands to KMC operations:

| Kimari Command | KMC Equivalent | Description |
|----------------|---------------|-------------|
| `kimari compress` | `kmc pack [--tensor-aware] [--codec] [--gguf-aware]` | Compress a model into a .kmc archive |
| `kimari compress-lora` | `kmc pack-lora` | Compress a LoRA adapter directory |
| `kimari compress-checkpoint` | `kmc pack-checkpoint` | Compress a training checkpoint directory |
| `kimari decompress` | `kmc unpack` | Decompress a .kmc archive |
| `kimari verify-compress` | `kmc verify` | Verify archive integrity |
| `kimari inspect-model` | `kmc inspect [--lora\|--checkpoint\|--gguf\|--tensors]` | Inspect a model directory or archive |
| `kimari bench-compress` | `kmc bench [--compare-zipnn] [--compare-codecs]` | Benchmark compression performance |

## Integration Layer

The adapter module is implemented at `src/kmc/integrations/kimari.py` and provides:

```python
from kmc.integrations.kimari import (
    kimari_compress,              # -> kmc pack [--tensor-aware] [--codec] [--gguf-aware]
    kimari_compress_lora,         # -> kmc pack-lora (alias: kimari_pack_lora)
    kimari_compress_checkpoint,   # -> kmc pack-checkpoint (alias: kimari_pack_checkpoint)
    kimari_decompress,            # -> kmc unpack
    kimari_verify_compress,       # -> kmc verify
    kimari_bench_compress,        # -> kmc bench [--compare-zipnn] [--compare-codecs]
    kimari_inspect_model,         # -> kmc inspect [--lora|--checkpoint|--gguf|--tensors]
    kimari_pack_lora,             # Direct alias for kimari_compress_lora
    kimari_pack_checkpoint,       # Direct alias for kimari_compress_checkpoint
    KIMARI_COMMAND_MAP,           # Command mapping dict
)
```

### Usage Examples

#### Compress a Model

```python
from kmc.integrations.kimari import kimari_compress

# Compress with tensor-aware mode (default for Kimari)
result = kimari_compress("./my-model", "./my-model.kmc", tensor_aware=True)
# result = {
#     "status": "ok",
#     "source": "./my-model",
#     "output": "./my-model.kmc",
#     "original_size": 500000000,
#     "compressed_size": 300000000,
#     "ratio": 0.6,
#     "tensor_aware": True,
#     "codec": "auto",
#     "gguf_aware": False,
# }

# Compress with GGUF-aware mode
result = kimari_compress("./model.gguf", "./model.kmc", gguf_aware=True)
# result includes gguf_aware: True
```

#### Compress a LoRA Adapter

```python
from kmc.integrations.kimari import kimari_pack_lora

result = kimari_pack_lora("./my-lora", "./my-lora.kmc")
# result = {
#     "status": "ok",
#     "artifact_type": "lora_adapter",
#     "peft_type": "LORA",
#     "lora_rank": 16,
#     "base_model": "meta-llama/Llama-2-7b-hf",
#     "target_modules": ["q_proj", "v_proj"],
#     "original_size": 51200000,
#     "compressed_size": 28300000,
#     "ratio": 0.5527,
# }

# If not a LoRA adapter
result = kimari_pack_lora("./not-a-lora", "./output.kmc")
# result = {
#     "status": "error",
#     "message": "Not a LoRA adapter directory: ./not-a-lora",
# }
```

#### Compress a Training Checkpoint

```python
from kmc.integrations.kimari import kimari_pack_checkpoint

result = kimari_pack_checkpoint("./checkpoint-1000", "./checkpoint-1000.kmc")
# result = {
#     "status": "ok",
#     "artifact_type": "training_checkpoint",
#     "step": 1000,
#     "has_optimizer_state": True,
#     "has_trainer_state": True,
#     "has_safetensors_model": False,
#     "original_size": 1500000000,
#     "compressed_size": 980000000,
#     "ratio": 0.6533,
# }
```

#### Verify an Archive

```python
from kmc.integrations.kimari import kimari_verify_compress

result = kimari_verify_compress("./my-model.kmc")
# result = {
#     "status": "ok",
#     "archive": "./my-model.kmc",
#     "integrity": "OK",
#     "errors": [],
#     "warnings": [],
#     "total_files": 5,
#     "total_blocks": 24,
# }
```

#### Inspect a Model

```python
from kmc.integrations.kimari import kimari_inspect_model

result = kimari_inspect_model("./my-model")
# result = {
#     "status": "ok",
#     "path": "./my-model",
#     "artifact_type": "huggingface_model",
#     "files": [
#         {
#             "path": "./my-model/model.safetensors",
#             "format": "safetensors",
#             "details": "safetensors: 291 tensors, 124M params, dtypes=[F32]",
#             "file_size": 501200000,
#         },
#         ...
#     ],
# }
```

### CLI Adapter Example

An example CLI adapter is provided at `examples/kimari_cli_adapter.py`:

```bash
python examples/kimari_cli_adapter.py compress ./model ./model.kmc
python examples/kimari_cli_adapter.py compress-lora ./lora ./lora.kmc
python examples/kimari_cli_adapter.py compress-checkpoint ./checkpoint ./checkpoint.kmc
python examples/kimari_cli_adapter.py decompress ./model.kmc ./restored
python examples/kimari_cli_adapter.py verify-compress ./model.kmc
python examples/kimari_cli_adapter.py bench-compress ./model ./model.kmc --compare-zipnn
python examples/kimari_cli_adapter.py inspect-model ./model
```

## Integration Architecture

```
+------------------------------------------+
|  Kimari CLI                              |
|                                          |
|  kimari compress ./model                 |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.integrations.kimari           | |  Adapter layer
|  |   kimari_compress()               | |
|  |   (tensor_aware=True)             | |
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.archive                        | |  Core KMC
|  |   pack(tensor_aware=True)          | |
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.formats.safetensors            | |  Format support
|  |   read_safetensors_info()          | |
|  +------------------------------------+ |
|                                          |
|  kimari compress-lora ./lora             |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.integrations.kimari           | |  Adapter layer
|  |   kimari_pack_lora()              | |
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.workflows.lora                 | |  LoRA workflow
|  |   detect_lora_adapter()            | |
|  |   build_lora_manifest_metadata()   | |
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.archive                        | |  Core KMC
|  |   pack(artifact_type="lora_adapter")| |
|  +------------------------------------+ |
|                                          |
|  kimari compress-checkpoint ./ckpt       |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.integrations.kimari           | |  Adapter layer
|  |   kimari_pack_checkpoint()        | |
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.workflows.checkpoint           | |  Checkpoint workflow
|  |   detect_checkpoint()              | |
|  |   build_checkpoint_manifest_metadata()|
|  +------------------------------------+ |
|       |                                  |
|  +------------------------------------+ |
|  | kmc.archive                        | |  Core KMC
|  |   pack(artifact_type="training_checkpoint")|
|  +------------------------------------+ |
+------------------------------------------+
```

## Design Principles

1. **KMC is independent**: The `kmc` CLI operates standalone. Kimari integration is optional.
2. **Adapter pattern**: The integration layer translates Kimari conventions to KMC calls without modifying KMC internals.
3. **Return values**: Integration functions return structured dicts or dataclasses, not CLI output.
4. **No circular dependencies**: KMC does not depend on Kimari. Kimari depends on KMC.
5. **Tensor-aware by default**: The Kimari adapter enables `tensor_aware=True` by default, since Kimari users are working with AI models and benefit from tensor-level metadata.
6. **Artifact-aware**: The adapter functions populate `artifact_type` and `artifact_metadata` in the manifest, enabling downstream tools to understand what was compressed.

## Phase 1: Adapter Layer (v0.3 -- Completed)

- [x] `kimari_compress()` adapter function with `tensor_aware` parameter
- [x] `kimari_decompress()` adapter function
- [x] `kimari_verify_compress()` adapter function
- [x] `kimari_bench_compress()` adapter function with `compare_zipnn` parameter
- [x] Command mapping documentation
- [x] CLI adapter example (`examples/kimari_cli_adapter.py`)

## Phase 1.5: Workflow Adapters (v0.5 -- Completed)

- [x] `kimari_pack_lora()` adapter function with LoRA detection and metadata
- [x] `kimari_pack_checkpoint()` adapter function with checkpoint detection and metadata
- [x] `kimari_inspect_model()` adapter function with artifact type detection
- [x] `gguf_aware` parameter on `kimari_compress()`
- [x] Updated `KIMARI_COMMAND_MAP` with new commands
- [x] Artifact type and metadata propagation through the adapter layer

## Phase 2: Kimari CLI Integration (v0.6)

- [ ] Add `kimari compress` subcommand to Kimari CLI
- [ ] Add `kimari decompress` subcommand
- [ ] Add `kimari verify-compress` subcommand
- [ ] Add `kimari bench-compress` subcommand
- [ ] Add `kimari compress-lora` subcommand
- [ ] Add `kimari compress-checkpoint` subcommand
- [ ] Shared configuration (block size, compression level)
- [ ] Progress reporting integration

## Phase 3: KimariDB Storage Backend

KMC archives can be stored in KimariDB with metadata indexing:

- Archive hash (SHA-256 of the .kmc file)
- Source model identifier (Hugging Face model ID, custom ID)
- Compression ratio and codec information
- Artifact type and artifact metadata
- Format metadata (safetensors dtypes, GGUF quantization summary)
- Tensor metadata summary
- Creation timestamp and tool version

This enables:
- Content-addressed storage (deduplication of identical archives)
- Fast lookup of compressed versions of known models
- Query by artifact type (find all LoRA adapter archives)
- Query by GGUF quantization type (find all Q4_K archives)
- Verification that stored archives have not been tampered with
- Tensor-level metadata queries (find archives containing BF16 tensors)

## Phase 4: Download-Cache-Verify Workflow

Integrate with Hugging Face Hub for a seamless workflow:

```bash
kimari download huggingface/gpt2 --compress --verify --tensor-aware
kimari download huggingface/my-lora --compress-lora --verify
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

> **Important:** Block-level serving does NOT reduce inference VRAM. The decompressed blocks still occupy the same memory. Runtime compressed loading (keeping blocks compressed in memory) is future research.

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
- Roundtrip via `kimari_pack_lora()` -> `kimari_decompress()`
- Roundtrip via `kimari_pack_checkpoint()` -> `kimari_decompress()`
- Verification via `kimari_verify_compress()` matches `kmc verify`
- Tensor-aware mode produces manifest with tensor entries
- LoRA adapter produces manifest with `artifact_type: "lora_adapter"`
- Checkpoint produces manifest with `artifact_type: "training_checkpoint"`
- GGUF-aware mode records format metadata
- ZipNN comparison works when zipnn is installed
- ZipNN gracefully degrades when zipnn is not installed
- Metadata consistency between KMC and KimariDB
- Backward compatibility of .kmc archives across KMC versions
- Error propagation from KMC to Kimari CLI
