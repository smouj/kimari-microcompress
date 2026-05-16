# Runtime Integration Research

> **Status**: Research document — not a feature specification
> **KMC version**: v0.8.0-alpha
> **Last updated**: 2025

---

> **⚠️ MANDATORY WARNING**
>
> **KMC v0.8 does not perform compressed inference. Runtime integration is research-only. No VRAM reduction is promised.**
>
> KMC is a lossless compression tool for model artifacts. It reduces disk storage and transfer size. It does NOT reduce VRAM usage during inference, it does NOT enable "compressed inference," and it does NOT modify the numerical behavior of models. Any integration with ML runtimes in the future would be for the purpose of on-demand decompression during model loading, not for operating on compressed data directly.

---

## Introduction

This document surveys how KMC's `.kmc` archive format could integrate with machine learning runtimes to improve the model loading experience. The central question is: **can a runtime read tensor data directly from a compressed `.kmc` archive without first extracting the entire archive to disk?**

The answer is nuanced. KMC already supports partial file access and tensor-level extraction via the `KMCReader` API. However, true "runtime integration" — where a model server or inference engine natively reads from `.kmc` archives — requires deeper integration work that is still in the research phase.

This document covers what is already working, what is being researched, and what is planned. All future features described here are aspirational and subject to change.

## Partial File Access (Already Working)

### Current Capabilities

As of KMC v0.7, the `.kmc` format supports partial file access through the `KMCReader` API. This is the foundation upon which all runtime integration is built. The key capabilities are:

**Block-level random access**: Every block in a `.kmc` archive has an `archive_offset` field that records its physical byte position in the archive file. This allows `KMCReader` to seek directly to a block's location and read only the compressed data for that block, decompress it, and return the original data — without reading or decompressing any other block in the archive.

**File-level extraction**: The `KMCReader.read_file(path)` method reads only the blocks belonging to a specific file, assembles them in order, and verifies the file's SHA-256 hash. This is efficient for extracting a single file from a large archive.

**Tensor-level extraction**: The `KMCReader.read_tensor(name)` method reads only the blocks belonging to a specific tensor. When the archive was created with `--tensor-aware`, blocks are aligned to tensor boundaries, and the tensor index maps tensor names to their block IDs. This allows reading a single tensor's raw bytes without decompressing the entire model.

**Byte-range extraction**: The `KMCReader.read_file_range(path, offset, length)` method reads a specific byte range from a file, decompressing only the blocks that overlap the requested range. This is useful for reading tensor metadata headers or small configuration sections without decompressing the full file.

### Performance Characteristics

Partial file access is I/O-bound rather than CPU-bound. The `KMCReader` performs the following operations for each block read:

1. Seek to `archive_offset` in the archive file (O(1))
2. Read `compressed_size` bytes (O(compressed_size))
3. Verify block SHA-256 hash (O(compressed_size))
4. Decompress using the recorded codec (O(original_size))

For zstd-compressed blocks, decompression throughput typically exceeds 1 GB/s on modern hardware, making partial access competitive with reading uncompressed files from disk for most use cases.

### Index Structures

The `KMCReader` builds three indexes on archive open:

- **BlockIndex**: Maps global block IDs to `BlockLocation` records (archive offset, compressed size, codec, hash, etc.)
- **FileIndex**: Maps file paths to `FileLocation` records (list of block IDs, SHA-256 hash, total size)
- **TensorIndex**: Maps tensor names to `TensorLocation` records (file path, block IDs, byte offset, byte size)

These indexes are built from the manifest, which is a JSON document at the beginning of the archive. Index construction is O(n) in the number of blocks, files, and tensors — typically negligible compared to I/O operations.

## Tensor-on-Demand Loading (Future)

### Concept

Tensor-on-demand loading is the idea that an ML runtime could load tensors from a `.kmc` archive one at a time, as needed, rather than decompressing the entire model into memory at startup. This would reduce peak memory usage during the loading phase and could enable loading models that are larger than available uncompressed disk space.

### Research Questions

1. **Latency overhead**: How much latency does per-tensor decompression add compared to reading from an uncompressed safetensors file? Initial estimates suggest 5–15% overhead for zstd-compressed blocks, but this needs benchmarking on real hardware and real models.

2. **Memory-mapped I/O**: Can `.kmc` archives be memory-mapped for efficient random access? The current format stores compressed blocks contiguously, which should be compatible with `mmap()`. However, the variable-length compressed blocks mean that tensor data is not at a fixed offset, requiring an index lookup before each read.

3. **Prefetching strategies**: Could a runtime prefetch tensors that are likely to be needed soon (e.g., the next layer's weights) while the current layer is executing? This would amortize decompression latency but requires integration with the runtime's execution scheduler.

4. **Concurrency**: Can multiple tensors be decompressed in parallel using a thread pool? The `KMCReader` is not currently thread-safe, but a concurrent reader could be implemented using per-thread file handles and decompression contexts.

### Technical Challenges

- **safetensors header parsing**: The safetensors format stores a JSON header at the beginning of the file with tensor metadata. KMC's tensor index captures this information during packing, but a runtime integration would need to parse the KMC manifest and map tensor names to archive offsets efficiently.

- **GPU direct loading**: Most ML runtimes load tensor data into GPU memory via `cudaMemcpy` or similar. Decompressing on the CPU and then copying to the GPU adds an extra hop compared to memory-mapped loading from an uncompressed file. Research is needed into whether GPU-direct storage (GDS) could be combined with KMC's block-level access.

- **Sharded model support**: Large models are typically stored across multiple safetensors shards. KMC packs all shards into a single `.kmc` archive, but the runtime must still understand the shard-to-tensor mapping. The manifest's `tensor_entries` provide this mapping, but the runtime integration layer must translate it.

## Integration with llama.cpp (Research)

### Background

llama.cpp is one of the most widely used inference engines for GGUF models. It reads `.gguf` files directly using memory-mapped I/O and supports quantized inference without decompression. KMC's value proposition for llama.cpp is different from its value for safetensors-based runtimes:

- **GGUF files are already compact**: Quantized GGUF files are much smaller than their FP32/FP16 safetensors counterparts. KMC's additional compression on top of quantized data is modest (typically 5–20% depending on the quantization level).
- **llama.cpp uses mmap extensively**: The GGUF format is designed for memory-mapped access, with tensor data at fixed offsets in the file. KMC's variable-length compressed blocks break this assumption.
- **Quantized inference is the norm**: llama.cpp operates directly on quantized data in memory. KMC cannot and should not change this — it would need to decompress blocks and present them to llama.cpp in the standard GGUF layout.

### Research Direction: KMC-backed GGUF Cache

One possible integration approach is to use KMC as a **storage and transfer cache** for GGUF files, rather than as a direct data source during inference:

1. **Download and cache**: Model files are distributed as `.kmc` archives (smaller than `.gguf` files). A model manager tool decompresses the `.gguf` file from the `.kmc` archive and stores it locally.
2. **On-demand extraction**: When llama.cpp needs a GGUF file, the KMC integration layer extracts it to a temporary location and provides the path to llama.cpp. The extracted file is cached for subsequent runs.
3. **Partial extraction for multi-model**: If only certain layers or tensors are needed (e.g., for model merging or layer-wise analysis), KMC's partial access can extract only the relevant portions.

This approach does not require any changes to llama.cpp itself. It treats KMC as a transparent compression layer for storage and distribution.

### Research Direction: Custom GGUF Provider

A more ambitious approach would be to implement a custom GGUF provider in llama.cpp that reads from `.kmc` archives:

1. Implement a `kmc_gguf_provider` that satisfies llama.cpp's GGUF provider interface.
2. The provider would use KMC's `KMCReader` to read tensor data on demand.
3. Read-ahead buffering would prefetch the next layer's tensors while the current layer is executing.

This approach is purely theoretical and has not been prototyped. Significant challenges include:

- llama.cpp's provider interface may not support asynchronous or on-demand tensor loading.
- The decompression latency per tensor may be unacceptable for real-time inference.
- Memory management becomes more complex when tensors are loaded on demand rather than mmap'd.

**Current status**: No prototype exists. This is a research area for future exploration.

## Integration with safetensors Loaders (Research)

### Background

The safetensors format is the standard for Hugging Face model distribution. Major frameworks (PyTorch, JAX, TensorFlow, Flax) can load safetensors files natively. KMC archives that contain safetensors files could integrate with these loaders to provide on-demand tensor access.

### Research Direction: KMCFilesystem Adapter

One approach is to implement a **virtual filesystem adapter** that presents a `.kmc` archive as a directory of safetensors files. The adapter would:

1. Parse the KMC manifest to enumerate files and their tensor metadata.
2. When a safetensors loader opens a "file" in the virtual filesystem, the adapter reads the safetensors header from the KMC archive.
3. When the loader requests tensor data, the adapter reads the relevant blocks from the KMC archive, decompresses them, and returns the tensor bytes.

This approach would work with any safetensors-compatible framework without modification. However, it requires:

- A FUSE (Filesystem in Userspace) implementation or similar virtual filesystem layer.
- Efficient caching of decompressed blocks to avoid redundant decompression.
- Thread safety for concurrent tensor reads.

### Research Direction: Direct KMC Reader for Hugging Face

A more targeted approach would be to implement a custom model loader for Hugging Face's `transformers` library that reads from `.kmc` archives directly:

```python
# Hypothetical future API
from kmc.integrations.huggingface import load_model_from_kmc

model = load_model_from_kmc("my-model.kmc", device="cuda")
# Internally: opens KMCReader, reads tensor metadata,
# decompresses tensors on demand, loads into PyTorch
```

This would require:

1. A `KMCModelLoader` class that implements Hugging Face's model loading protocol.
2. Mapping from KMC tensor names to PyTorch state dict keys.
3. Handling of sharded models (multiple safetensors files in one `.kmc` archive).
4. Support for `torch.dtype` conversion during loading.

**Current status**: No prototype exists. The `KMCReader` API provides the foundation, but the Hugging Face integration layer has not been built.

## Integration with Kimari Runtime (Planned)

### Background

The **Kimari runtime** is a planned ML inference and model management runtime that will use KMC as its native storage format. Unlike the integrations above (which adapt KMC to existing runtimes), the Kimari runtime is designed from the ground up to work with `.kmc` archives.

### Planned Features

1. **Native KMC reader**: The Kimari runtime will use KMC's `KMCReader` API directly for all model I/O. No extraction to intermediate files is needed.

2. **Tensor streaming**: The runtime will support streaming tensor data from `.kmc` archives, loading tensors on demand as inference progresses through the model's layers.

3. **Model package registry**: A registry of KMC-packaged models with metadata (artifact type, format, quantization level, tensor dtypes, shapes). The registry enables model discovery and versioning.

4. **Transparent decompression**: The runtime will handle decompression internally, presenting a simple API to the user (e.g., `kimari serve my-model.kmc`).

5. **Cache management**: Frequently accessed tensors will be cached in decompressed form in memory or on a fast local disk, reducing latency for repeated inference requests.

### Current Status

The Kimari runtime integration is in the planning phase. The `kmc.integrations.kimari` module provides the adapter layer that maps Kimari CLI commands to KMC operations:

| Kimari Command | KMC Operation |
|---|---|
| `kimari compress` | `kmc pack [--tensor-aware] [--codec]` |
| `kimari compress-lora` | `kmc pack-lora` |
| `kimari compress-checkpoint` | `kmc pack-checkpoint` |
| `kimari decompress` | `kmc unpack` |
| `kimari verify-compress` | `kmc verify` |
| `kimari inspect-model` | `kmc inspect [--lora\|--checkpoint\|--gguf\|--tensors]` |
| `kimari bench-compress` | `kmc bench [--compare-zipnn] [--compare-codecs]` |

These commands provide the storage and management layer. The inference runtime layer is not yet implemented.

### Runtime Hints

KMC v0.8 archives include a `runtime_hints` field in the manifest that provides information about the archive's capabilities for runtime consumers:

```json
{
  "runtime_hints": {
    "partial_file_access": "supported",
    "tensor_access": "index_based",
    "compressed_inference": false
  }
}
```

These hints allow the Kimari runtime (or any other consumer) to determine what operations are supported without parsing the entire manifest.

## Summary and Disclaimers

| Feature | Status | Notes |
|---|---|---|
| Partial file access | **Working** | Available via `KMCReader` since v0.7 |
| Tensor-level extraction | **Working** | Available via `KMCReader.read_tensor()` since v0.7 |
| Byte-range extraction | **Working** | Available via `KMCReader.read_file_range()` since v0.7 |
| Tensor-on-demand loading | **Research** | Not implemented; latency and concurrency under study |
| llama.cpp integration | **Research** | No prototype; cache-based approach most feasible |
| safetensors loader integration | **Research** | No prototype; FUSE adapter or custom loader under consideration |
| Kimari runtime integration | **Planned** | CLI adapter exists; inference runtime not yet built |

---

> **⚠️ REMINDER**
>
> **KMC v0.8 does not perform compressed inference. Runtime integration is research-only. No VRAM reduction is promised.**
>
> KMC reduces disk storage and transfer size. Any future runtime integration would be for the purpose of on-demand decompression during model loading, not for operating on compressed data during inference. Claims that KMC or any similar tool can reduce VRAM usage during inference are incorrect and should not be made.
