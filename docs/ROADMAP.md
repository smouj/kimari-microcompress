# Roadmap

This document tracks the development progress and future plans for Kimari MicroCompress. Each version represents a milestone with specific goals and deliverables.

## Known Limitations (Current)

These are **real, documented limitations** of the current KMC implementation. They are not bugs — they are scope boundaries:

1. **KMC does NOT reduce VRAM during inference.** Compressed archives must be fully unpacked before a model can be loaded. Block-loading is future research.
2. **KMC does NOT modify model weights.** Compression is strictly lossless. No quality degradation occurs at any point.
3. **Block-loading is not implemented.** The manifest contains per-block offsets, but on-demand decompression of individual blocks is a future feature.
4. **GGUF-aware compression is experimental.** The `--gguf-aware` flag adjusts codec selection for quantized tensors, but does not yet implement block-level GGUF-specific compression strategies.
5. **No fixed compression ratios.** Results depend heavily on model format, data type, and content. Synthetic benchmarks produce misleadingly high ratios.
6. **KMC is not quantization.** Use quantization (GGUF Q4_K, GPTQ, AWQ) for smaller inference models. KMC is complementary: it compresses files for storage and transfer.
7. **No pickle deserialization.** KMC never loads pickle-based files (optimizer.pt, training_args.bin, pytorch_model.bin). These are compressed as raw bytes with size/hash recorded.
8. **LoRA delta compression is not yet implemented.** LoRA adapters are compressed using standard or tensor-aware mode. Delta compression relative to a base model is future work.
9. **KMC is lossless only.** There is no lossy mode and no weight modification of any kind.
10. **No streaming I/O.** The entire archive must be loaded into memory for operations. Streaming pack/unpack for minimal memory footprint is planned for v0.6.
11. **No parallel compression.** Blocks are compressed sequentially. Parallel block compression is planned for v0.6.

---

## v0.1 — MVP Archive ✅ Completed

The initial release establishes the core infrastructure:

- [x] `.kmc` archive format with JSON manifest
- [x] `kmc pack`, `kmc unpack`, `kmc verify`, `kmc inspect`, `kmc bench` CLI commands
- [x] zstd / zlib / raw codec selection per-block
- [x] SHA-256 integrity verification at file and block level
- [x] 256 KiB micro-blocks (configurable)
- [x] AI model format detection (safetensors, GGUF, .bin, .pt, .ckpt)
- [x] safetensors tensor metadata parsing
- [x] GGUF header parsing
- [x] Path traversal protection in `unpack()`
- [x] Basic roundtrip and manifest tests
- [x] CI with GitHub Actions (pytest + ruff)

## v0.2 — Security + Verification + Benchmark ✅ Completed

Focus on security, robustness, and realistic testing:

- [x] Harden `unpack()` with `safe_join_extract_path()` function
- [x] Comprehensive path traversal tests
- [x] Manifest size limits to prevent DoS (100 MB max)
- [x] Manifest validation: duplicate paths, unsupported codecs, size mismatches
- [x] Full verification report (`verify_full()`) with structured output
- [x] Block and file hash verification in verify
- [x] Symlink overwrite protection in unpack
- [x] Realistic benchmark system with codec comparison
- [x] Benchmark JSON output and file export
- [x] Synthetic data flag for benchmarks
- [x] Enhanced inspector: LoRA adapters, tokenizer files, config files, model shards
- [x] safetensors real tensor metadata (names, shapes, dtypes, param counts)
- [x] GGUF header: version, tensor count, metadata KV count
- [x] Kimari CLI integration adapters
- [x] Documented limitations in README and ROADMAP

## v0.3 — Safetensors + ZipNN Comparison + GGUF Parser ✅ Completed

Real safetensors support, ZipNN benchmark comparison, and minimal GGUF parser:

- [x] Dedicated `src/kmc/formats/safetensors.py` module
- [x] Read safetensors header without loading weights
- [x] Extract tensor metadata: name, dtype, shape, offsets, byte size
- [x] Detect sharded models (`model-NNNN-of-MMMM.safetensors`)
- [x] Detect LoRA/PEFT adapters with rank and target modules
- [x] Graceful degradation when `safetensors` package is not installed
- [x] Tensor-aware packing mode (`--tensor-aware`)
- [x] Tensor entries in manifest (v2 manifest format)
- [x] Backward compatibility with v0.2 `.kmc` archives
- [x] `kmc inspect --json` and `kmc inspect --tensors` CLI flags
- [x] Directory-level model inspection (detected type, sharding, LoRA, etc.)
- [x] ZipNN benchmark comparison (`--compare-zipnn`)
- [x] Environment metadata in benchmark output
- [x] No invented benchmarks or superiority claims
- [x] Dedicated GGUF module with endianness detection
- [x] Hugging Face workflow documentation

## v0.4 — Tensor-Aware Block Codecs ✅ Completed

Real tensor-specific compression algorithms:

- [x] BF16/FP16/FP32 byte-plane separation codec (BytePlane)
- [x] BF16/FP16/FP32 sign/exponent/mantissa separation codec (FloatPlane)
- [x] Per-dtype compression strategies (dtype-based candidate chains)
- [x] Block-level codec selection based on tensor dtype (automatic selector)
- [x] Per-block codec metadata in manifest (v3 format)
- [x] `--codec auto|byteplane|floatplane|zstd|zlib|raw` CLI flag
- [x] `--compression` flag for `kmc inspect`
- [x] `--compare-codecs` flag for `kmc bench`
- [x] Benchmark tensor-aware codecs against generic compression
- [x] `scripts/bench_small_hf_model.py` for real model benchmarks
- [x] 161 tests passing, ruff clean

## v0.5 — GGUF Metadata + LoRA/Checkpoint Workflows ✅ Completed

Full GGUF tensor metadata parsing, specialized workflows, experimental GGUF-aware compression:

- [x] GGUF tensor metadata parsing (full tensor descriptors: name, shape, type, offset, size)
- [x] GGUF quantization summary (e.g., Q4_K: 201, F32: 1)
- [x] Experimental `--gguf-aware` compression mode
- [x] Manifest v4 with `artifact_type`, `artifact_metadata`, `format_metadata`
- [x] Artifact auto-detection (huggingface_model, gguf_model, lora_adapter, training_checkpoint, unknown)
- [x] LoRA/PEFT adapter workflow: `kmc pack-lora` and `kmc inspect --lora`
- [x] Training checkpoint workflow: `kmc pack-checkpoint` and `kmc inspect --checkpoint`
- [x] GGUF inspection workflow: `kmc inspect --gguf` with tensor details
- [x] Real small-model benchmark script (`scripts/bench_small_hf_model.py`)
- [x] Kimari CLI integration preparation (compress-lora, compress-checkpoint, inspect-model)
- [x] `workflows/` subpackage with `lora.py` and `checkpoint.py`
- [x] Backward compatible with .kmc v0.2/v0.3/v0.4
- [x] 228 tests passing, ruff clean
- [x] CONTRIBUTING.md and CHANGELOG.md added
- [x] Enhanced CI with lint + format + CLI verification

---

## v0.6 — Kimari CLI + Streaming + Parallel (Next)

Full integration with the Kimari ecosystem and performance improvements. This version focuses on production readiness: performance, reliability, and seamless integration.

### Kimari CLI Integration

- [ ] `kimari compress` command in Kimari CLI
- [ ] `kimari decompress` command
- [ ] `kimari verify-compress` command
- [ ] `kimari bench-compress` command
- [ ] `kimari compress-lora` command
- [ ] `kimari compress-checkpoint` command
- [ ] Shared configuration (block size, compression level)
- [ ] Progress reporting integration with Kimari UI
- [ ] KimariDB storage backend integration
- [ ] Content-addressed archive storage

### Streaming I/O

- [ ] Streaming pack: process files without loading entire archive into memory
- [ ] Streaming unpack: write blocks incrementally without full archive buffering
- [ ] Memory-mapped manifest reading for large archives
- [ ] Progress callbacks for long-running operations
- [ ] Estimated time remaining based on throughput

### Parallel Compression

- [ ] Parallel block compression using `concurrent.futures` or `multiprocessing`
- [ ] Configurable worker count (default: number of CPU cores)
- [ ] Thread-safe codec operations (ensure all codecs are reentrant)
- [ ] Parallel block decompression for faster unpack
- [ ] Benchmark parallel vs sequential performance

### Reliability

- [ ] Resume interrupted pack operations (checkpoint file)
- [ ] Resume interrupted unpack operations
- [ ] Archive corruption detection and partial recovery
- [ ] Comprehensive error messages with actionable suggestions
- [ ] Logging framework integration (structured logging)

## v0.7 — Experimental Loader/Runtime Work

Block-level loading and runtime integration. This is the most ambitious milestone, enabling KMC to serve as a model loading backend rather than just a storage format.

### Block-Level Loading

- [ ] Block-level loading (partial decompression on demand)
- [ ] `BlockServer` class for programmatic block access
- [ ] Load specific tensors by name without decompressing the entire archive
- [ ] Load tensor ranges (e.g., first N layers)
- [ ] Memory-mapped archive reading for large files

### Runtime Integration

- [ ] Python API for programmatic block access
- [ ] Integration with Hugging Face `from_pretrained()` loading
- [ ] Integration with llama.cpp GGUF loading
- [ ] Block server for remote block fetching (HTTP/gRPC)
- [ ] Runtime compressed loading (research phase — keeping blocks compressed in memory)

### Performance

- [ ] Lazy tensor loading (only decompress when first accessed)
- [ ] Tensor caching (keep recently used tensors decompressed)
- [ ] Prefetching (anticipate which tensors will be needed next)
- [ ] Benchmarks for block-level vs full-archive loading

> **Important:** Block-level loading does NOT reduce inference VRAM. The decompressed tensors still occupy the same GPU memory. Runtime compressed loading (keeping blocks compressed in CPU RAM) is future research.

## v0.8 — Advanced Compression

More sophisticated compression strategies for specific use cases:

### Delta Compression

- [ ] Delta compression for LoRA adapters relative to base models
- [ ] Delta compression between training checkpoints
- [ ] XOR delta encoding for adjacent weight matrix rows
- [ ] Delta manifest schema (v5 format)

### GGUF-Specific

- [ ] GGUF block-level compression (skip already-quantized blocks entirely)
- [ ] GGUF metadata section compression (highly compressible KV pairs)
- [ ] GGUF vocabulary/tokenizer data optimization

### Optimizer State

- [ ] Optimizer state compression for training checkpoints
- [ ] Adam state separation (momentum, variance, step count)
- [ ] Sparse optimizer state handling

### Cross-File Optimization

- [ ] Weight sharing detection across model files
- [ ] Cross-file deduplication (identical blocks stored once)
- [ ] Dictionary-trained zstd for similar tensor blocks
- [ ] Cross-model compression (shared vocabulary, shared embeddings)

## v0.9 — Encryption and Authentication

Security features for production deployments:

- [ ] Symmetric encryption of archive contents (AES-256-GCM)
- [ ] Asymmetric encryption (recipient-specific archives)
- [ ] Archive signing (Ed25519 or RSA signatures)
- [ ] Signature verification in `kmc verify`
- [ ] Key management integration (environment variables, key files, key servers)
- [ ] Encrypted manifest with selective disclosure

## v1.0 — Production Release

The first stable release with all major features and documented behavior:

- [ ] Stable API guarantee (no breaking changes without major version bump)
- [ ] Complete documentation coverage
- [ ] Performance benchmarks against competing tools
- [ ] Security audit (formal or community)
- [ ] Full Kimari ecosystem integration
- [ ] Python 3.10-3.14 support matrix tested
- [ ] Windows, macOS, Linux support tested
- [ ] Migration guide from v0.x to v1.0

---

## Future Research Directions

These are exploratory areas that may or may not be incorporated into KMC:

- **Neural compression**: Using small neural networks to compress weight matrices more efficiently than general-purpose codecs. Research suggests that learned compression can outperform hand-crafted codecs on specific data distributions.

- **Quantization-aware compression**: Exploiting the structure of quantized weights (e.g., GGUF quantization levels, GPTQ group sizes) for better compression. Quantized data has specific statistical properties that general-purpose codecs don't exploit.

- **Distributed compression**: Coordinating compression across multiple nodes for federated learning scenarios. Each node compresses its local model, and the archives are merged or compared centrally.

- **Incremental archives**: Supporting efficient updates to existing archives without full repack. Only changed blocks would need to be recompressed and the manifest updated.

- **Compressed KV cache**: Investigating whether compression techniques can reduce the memory footprint of KV caches during inference. This is distinct from model weight compression and requires runtime integration.

- **Weight pruning integration**: Detecting and exploiting sparse weight patterns (zeros, near-zeros) for better compression. Sparse tensors can be stored more efficiently by separating indices from values.

- **Model merging**: Supporting efficient storage of merged models (e.g., model soup, task vectors) by storing only the differences from base models.
