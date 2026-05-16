# Kimari Runtime Plan

> **Status**: Planning document — not a commitment to specific features or timelines
> **KMC version**: v0.8.0-alpha
> **Last updated**: 2025

---

> **⚠️ IMPORTANT DISCLAIMER**
>
> **Compressed inference is NOT supported. KMC is a lossless storage compression tool. It does not reduce VRAM usage, and the Kimari runtime plan does not include "compressed inference" as a feature.** All runtime integration work is focused on efficient model loading from `.kmc` archives — the model is decompressed into its native format in memory before any computation occurs.

---

## Current State

### What Exists Today

As of KMC v0.8.0-alpha, the following components are available for runtime integration:

**KMC Core Library**: The `kmc` Python package provides a complete compression, verification, and inspection pipeline. Key modules include:

- `kmc.archive` — Pack, unpack, verify, and inspect `.kmc` archives
- `kmc.reader` — `KMCReader` partial-access API with block, file, and tensor indexes
- `kmc.manifest` — Manifest data model (v7) with support for tensor metadata, dedup, delta, and runtime hints
- `kmc.codecs` — Codec library (zstd, zlib, byteplane, floatplane, gguf_quant_block, raw) with automatic selection
- `kmc.index` — Block, file, and tensor index structures for random access
- `kmc.dedup` — Cross-file deduplication (experimental)
- `kmc.delta` — Delta compression against base archives (experimental)
- `kmc.workflows` — Specialized workflows for LoRA adapters and training checkpoints

**Kimari CLI Adapter**: The `kmc.integrations.kimari` module maps Kimari CLI commands to KMC operations:

```
kimari compress           → kmc pack [--tensor-aware] [--codec]
kimari compress-lora      → kmc pack-lora
kimari compress-checkpoint → kmc pack-checkpoint
kimari decompress         → kmc unpack
kimari verify-compress    → kmc verify
kimari inspect-model      → kmc inspect
kimari bench-compress     → kmc bench
```

**Kimari Plugin Registration**: The `kmc.integrations.kimari_plugin` module provides a plugin interface for registering KMC as a compression backend within the Kimari ecosystem. This allows Kimari to discover and use KMC without hard-coding the integration.

**KMCReader API**: The partial-access reader supports:

- `list_files()` / `list_tensors()` — Enumerate archive contents
- `read_file(path)` — Read a complete file
- `read_tensor(name)` — Read a specific tensor's raw bytes
- `read_file_range(path, offset, length)` — Read a byte range
- `extract_file(path, output_dir)` — Extract a file to disk
- `extract_tensor(name, output_dir)` — Extract a tensor to disk

### What Does NOT Exist

- **Kimari inference runtime**: There is no runtime that can serve models directly from `.kmc` archives.
- **Tensor streaming**: Tensors cannot be streamed progressively from a `.kmc` archive during inference.
- **Model registry**: There is no centralized registry of KMC-packaged models.
- **GPU integration**: KMC has no GPU-side components; all decompression happens on the CPU.

## Short-term Goals: Index-Based Partial Loading

### Objective

Enable the Kimari runtime to load specific tensors from a `.kmc` archive without extracting the entire archive to disk, using the existing `KMCReader` API.

### Design

The short-term goal is to build a **KimariModelLoader** class that bridges KMC's `KMCReader` with common model loading patterns:

```python
# Target API (not yet implemented)
from kimari.runtime import KimariModelLoader

loader = KimariModelLoader("my-model.kmc")

# List available tensors
tensor_names = loader.list_tensors()  # ["model.layers.0.weight", ...]

# Load a specific tensor into a PyTorch tensor
weight = loader.load_tensor("model.layers.0.weight", device="cuda:0")

# Load all tensors for a specific file
state_dict = loader.load_file("model-00001-of-00005.safetensors")

# Load the full model (progressive, tensor-by-tensor)
model = loader.load_model(device="cuda:0")
```

### Implementation Steps

1. **Tensor name normalization**: KMC stores tensor names as they appear in the source format (safetensors, GGUF). The loader must map these to the framework's naming convention (e.g., PyTorch state dict keys may differ from safetensors keys in edge cases).

2. **Dtype conversion**: `KMCReader.read_tensor()` returns raw bytes. The loader must interpret these bytes according to the tensor's dtype (BF16, FP16, FP32, etc.) and convert to the target framework's tensor type. This requires:
   - Reading `tensor_dtype` and `tensor_shape` from the manifest's `TensorEntry`
   - Constructing a NumPy or PyTorch tensor from the raw bytes with the correct dtype and shape
   - Optionally converting to a different dtype (e.g., BF16 → FP32 for computation)

3. **Sharded model handling**: Large models are stored across multiple safetensors shards. The loader must:
   - Detect sharded models from the manifest (multiple files with `model-XXXXX-of-YYYYY.safetensors` naming pattern)
   - Read the shard index (typically `model.safetensors.index.json`) to map tensor names to shards
   - Load tensors from the appropriate shard on demand

4. **Memory management**: For large models, loading all tensors simultaneously may exceed available RAM. The loader should support:
   - **Progressive loading**: Load tensors one at a time and immediately transfer to GPU
   - **LRU cache**: Keep recently accessed tensors in a decompressed cache for repeated access
   - **Memory budget**: Set a maximum memory budget for the decompressed cache

5. **Configuration file handling**: Model directories contain configuration files (`config.json`, `tokenizer.json`, etc.) that are needed for inference. The loader must:
   - Read and parse these files from the `.kmc` archive
   - Make them available to the framework's model constructor

### Performance Targets

- **First-tensor latency**: < 2 seconds for opening the archive and reading the first tensor (dominated by index construction and first block decompression)
- **Per-tensor latency**: < 50 ms for reading a 10 MB tensor from an already-open archive (dominated by decompression)
- **Full model load time**: No more than 1.5× the time to load from an uncompressed safetensors file (the overhead comes from per-block decompression and checksum verification)

### Dedup and Delta Handling

When the `.kmc` archive uses deduplication (`--dedup`) or delta compression (`--delta-base`), the loader must handle these transparently:

- **Dedup**: When reading a tensor that contains deduplicated blocks, resolve `dedup_ref` to the canonical block and read from there.
- **Delta**: When reading a delta archive, load referenced blocks from the base archive. This requires the base archive to be accessible.

The `KMCReader` already handles these cases internally, so the loader does not need special logic — it just uses `KMCReader.read_tensor()`.

## Medium-term Goals: Tensor Streaming from .kmc

### Objective

Enable progressive tensor loading where the Kimari runtime streams tensors from a `.kmc` archive as inference proceeds, rather than loading all tensors before inference begins.

### Concept

In a typical transformer model, inference processes one layer at a time. The weights for layer N are needed only when the forward pass reaches layer N. Tensor streaming exploits this by loading layer N's weights just before they are needed and releasing them after layer N's computation is complete.

This is not "compressed inference" — the tensors are fully decompressed in GPU memory during computation. The benefit is **reduced peak CPU RAM usage during loading** and **faster time-to-first-token** for large models, since the model doesn't need to be fully loaded before inference can begin.

### Design Sketch

```python
# Conceptual API (not yet implemented)
from kimari.runtime import KimariStreamingLoader

loader = KimariStreamingLoader("large-model.kmc", device="cuda:0")

# Start inference with streaming
with loader.stream() as stream:
    for layer_idx in range(model.config.num_layers):
        # This triggers loading and decompression of layer weights
        layer_weights = stream.get_layer(layer_idx)
        # ... perform computation ...
        # After computation, weights can be released from CPU RAM
        stream.release_layer(layer_idx)
```

### Technical Requirements

1. **Layer-aware indexing**: The loader must understand the model's layer structure and map layer indices to tensor names. This requires either:
   - Parsing `config.json` from the archive to determine the layer structure
   - Heuristic matching of tensor name patterns (e.g., `model.layers.{i}.*`)
   - Explicit layer mapping provided by the user

2. **Asynchronous decompression**: To hide decompression latency, the loader should decompress the next layer's tensors while the current layer is being computed. This requires:
   - A background thread pool for decompression
   - A prefetch buffer that holds decompressed tensors waiting to be transferred to GPU
   - Synchronization between the inference thread and the decompression threads

3. **GPU transfer pipelining**: After decompression, tensor data must be copied from CPU to GPU memory. This transfer should be pipelined with decompression:
   - While layer N is being computed on GPU, layer N+1's tensors are being decompressed on CPU
   - As soon as decompression completes, the tensors are transferred to GPU
   - By the time layer N's computation finishes, layer N+1's tensors are ready on GPU

4. **Memory pressure management**: The streaming loader must respect memory limits:
   - **CPU RAM budget**: Maximum memory for decompressed tensors waiting for GPU transfer
   - **GPU VRAM budget**: Maximum memory for tensors on GPU (though this is managed by the inference engine)
   - **Backpressure**: If the prefetch buffer is full, the decompression thread should pause

### Challenges

- **Non-sequential access**: Some inference patterns (e.g., speculation, beam search) may access layers out of order, defeating the prefetch strategy.
- **KV cache interaction**: The KV cache grows during inference and competes with model weights for GPU memory. The streaming loader must coordinate with the inference engine's memory manager.
- **Cross-layer tensors**: Some tensors (e.g., embedding tables, layer norm scales) are accessed by every layer and should be pinned in memory rather than streamed.
- **Thread safety**: `KMCReader` is not currently thread-safe. The streaming loader would need to either use separate `KMCReader` instances per thread or add locking to `KMCReader`.

## Long-term Goals: Kimari Model Package Registry

### Objective

Create a registry of KMC-packaged models that enables discovery, versioning, and efficient distribution of ML models in the `.kmc` format.

### Concept

The Kimari model package registry would function similarly to a package manager (like PyPI or npm) but for ML models:

```
kimari publish my-model.kmc --name "org/llama-7b-chat" --version "1.0.0"
kimari search "llama-7b"
kimari install "org/llama-7b-chat@1.0.0"
kimari serve "org/llama-7b-chat@1.0.0"
```

### Registry Metadata

Each published model would include:

- **Model identity**: Name, version, author, license
- **Format metadata**: Artifact type (safetensors, GGUF, LoRA, checkpoint), quantization level, tensor dtypes, shapes
- **Compression metadata**: Codec used, compression ratio, block size, dedup/delta info
- **Runtime hints**: Supported runtimes, minimum hardware requirements, recommended configuration
- **Integrity**: SHA-256 hash of the archive, manifest hash
- **Provenance**: Base model reference (for LoRA adapters and fine-tuned models), training configuration

### Delta-Based Distribution

For model updates (e.g., a new fine-tuning step or a minor version update), the registry would support delta-based distribution:

1. The publisher creates a delta archive against the previous version using `--delta-base`
2. The registry stores the delta archive alongside the base version
3. When a user who already has the base version requests the update, only the delta archive is transferred
4. The user's local Kimari runtime reconstructs the full archive from the base + delta

This approach can reduce download sizes by 90% or more for incremental updates.

### Security Model

The registry would enforce:

- **Archive integrity verification**: All archives are verified against their recorded SHA-256 hash before being made available
- **Manifest validation**: Manifests are validated against the schema before publication
- **Delta chain integrity**: Delta archives include the base archive's SHA-256, preventing tampering with the base
- **Reproducibility**: All compression parameters (codec, level, block size) are recorded in the manifest, enabling reproducible archive creation

### Challenges

- **Registry infrastructure**: Building and maintaining a model registry requires significant infrastructure (storage, CDN, metadata database, API).
- **Model licensing**: Different models have different licenses. The registry must track and enforce license terms.
- **Large file distribution**: Even compressed, large models (100+ GB) are expensive to distribute. CDN costs and bandwidth limitations are significant.
- **Backward compatibility**: The registry must support archives created with different KMC versions, including future versions with different manifest formats.

---

## Appendix: Runtime Hints in the Manifest

KMC v0.8 archives include a `runtime_hints` field that provides machine-readable information about the archive's capabilities:

```json
{
  "runtime_hints": {
    "partial_file_access": "supported",
    "tensor_access": "index_based",
    "compressed_inference": false
  }
}
```

| Hint | Values | Description |
|---|---|---|
| `partial_file_access` | `"supported"`, `"unsupported"` | Whether the archive supports partial file reads (always `"supported"` for v0.7+ archives with block offsets) |
| `tensor_access` | `"index_based"`, `"none"` | Whether the archive has a tensor index for direct tensor access |
| `compressed_inference` | `false` | Always `false` — KMC does not support compressed inference |

These hints allow the Kimari runtime (or any consumer) to determine what operations are possible without parsing the entire manifest.

---

> **⚠️ FINAL REMINDER**
>
> None of the goals described in this document include compressed inference. KMC is a storage compression tool. The Kimari runtime plan focuses on efficient loading from compressed archives, not on performing computation on compressed data. Any claims that KMC reduces VRAM usage during inference are incorrect.
