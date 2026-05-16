# Roadmap

## Known Limitations (Current)

These are **real, documented limitations** of the current KMC implementation. They are not bugs -- they are scope boundaries:

1. **KMC does NOT reduce VRAM during inference.** Compressed archives must be fully unpacked before a model can be loaded. Block-loading is future research.
2. **KMC does NOT modify model weights.** Compression is strictly lossless. No quality degradation occurs at any point.
3. **Block-loading is not implemented.** The manifest contains per-block offsets, but on-demand decompression of individual blocks is a future feature.
4. **GGUF-aware compression is experimental.** The `--gguf-aware` flag adjusts codec selection for quantized tensors, but does not yet implement block-level GGUF-specific compression strategies (e.g., skipping already-quantized blocks entirely).
5. **No fixed compression ratios.** Results depend heavily on model format, data type, and content. Synthetic benchmarks produce misleadingly high ratios and should not be used as references.
6. **KMC is not quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses files for storage and transfer.
7. **No pickle deserialization.** KMC never loads pickle-based files (optimizer.pt, training_args.bin, pytorch_model.bin). These are compressed as raw bytes only, with size and hash recorded.
8. **LoRA delta compression is not yet implemented.** LoRA adapters are compressed using standard or tensor-aware mode. Delta compression relative to a base model is future work.
9. **KMC is lossless only.** There is no lossy mode and no weight modification of any kind.

## v0.1 -- MVP Archive

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

## v0.2 -- Security + Verification + Benchmark

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

## v0.3 -- Safetensors + ZipNN Comparison + GGUF Parser

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
- [x] Environment metadata in benchmark output (Python, OS, CPU, RAM, versions)
- [x] No invented benchmarks or superiority claims
- [x] Dedicated `src/kmc/formats/gguf.py` module with endianness detection
- [x] GGUF header: magic, version, endianness, tensor count, KV count, file size
- [x] Synthetic GGUF test files
- [x] Kimari CLI adapter with `tensor_aware` and `compare_zipnn` support
- [x] Hugging Face workflow documentation
- [x] Updated README, ROADMAP, ARCHITECTURE, BENCHMARK_PLAN

## v0.4 -- Tensor-Aware Block Codecs (Completed)

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

> **Note:** KMC v0.4 does NOT reduce inference VRAM. It compresses model storage and transfer artifacts. Runtime compressed loading remains future work.

## v0.5 -- GGUF Metadata + LoRA Workflows + Real Benchmarks (Completed)

Full GGUF tensor metadata parsing, specialized workflows for LoRA adapters and training checkpoints, experimental GGUF-aware compression, and real model benchmarks:

- [x] GGUF tensor metadata parsing (full tensor descriptors: name, shape, type, offset, size)
- [x] GGUF quantization summary (e.g., Q4_K: 201, F32: 1)
- [x] Experimental `--gguf-aware` compression mode
- [x] Manifest v4 with `artifact_type`, `artifact_metadata`, `format_metadata`
- [x] Artifact auto-detection (`huggingface_model|gguf_model|lora_adapter|training_checkpoint|unknown`)
- [x] LoRA/PEFT adapter workflow: `kmc pack-lora` and `kmc inspect --lora`
- [x] Training checkpoint workflow: `kmc pack-checkpoint` and `kmc inspect --checkpoint`
- [x] GGUF inspection workflow: `kmc inspect --gguf` with tensor details
- [x] Real small-model benchmark script (`scripts/bench_small_hf_model.py`)
- [x] Kimari CLI integration preparation (compress-lora, compress-checkpoint, inspect-model)
- [x] `workflows/` subpackage with `lora.py` and `checkpoint.py`
- [x] Backward compatible with .kmc v0.2/v0.3/v0.4
- [x] 228 tests passing, ruff clean

> **Note:** KMC v0.5 does NOT reduce inference VRAM. The `--gguf-aware` flag is experimental and may change in future releases. No pickle is used at any point.

## v0.6 -- Kimari CLI + Streaming + Parallel

Full integration with the Kimari ecosystem and performance improvements:

- [ ] `kimari compress` command in Kimari CLI
- [ ] `kimari decompress` command
- [ ] `kimari verify-compress` command
- [ ] `kimari bench-compress` command
- [ ] `kimari compress-lora` command
- [ ] `kimari compress-checkpoint` command
- [ ] Shared configuration (block size, compression level)
- [ ] Progress reporting integration
- [ ] KimariDB storage backend integration
- [ ] Parallel block compression/decompression
- [ ] Streaming pack/unpack for minimal memory footprint

## v0.7 -- Experimental Loader/Runtime Work

Block-level loading and runtime integration:

- [ ] Block-level loading (partial decompression on demand)
- [ ] Block server for remote block fetching
- [ ] Memory-mapped archive reading for large files
- [ ] Python API for programmatic block access
- [ ] Integration with model loading frameworks
- [ ] Runtime compressed loading (research phase)

## v0.8 -- Advanced Compression

More sophisticated compression strategies:

- [ ] Delta compression for LoRA adapters relative to base models
- [ ] GGUF block-level compression (skip already-quantized blocks entirely)
- [ ] Optimizer state compression for training checkpoints
- [ ] Weight sharing detection across model files
- [ ] Cross-file deduplication
- [ ] Dictionary-trained zstd for similar tensor blocks
- [ ] XOR delta encoding for adjacent weight matrix rows

## Future Research Directions

These are exploratory areas that may or may not be incorporated:

- **Neural compression**: Using small neural networks to compress weight matrices more efficiently than general-purpose codecs.
- **Quantization-aware compression**: Exploiting the structure of quantized weights (e.g., GGUF quantization levels) for better compression.
- **Distributed compression**: Coordinating compression across multiple nodes for federated learning scenarios.
- **Incremental archives**: Supporting efficient updates to existing archives without full repack.
- **Compressed KV cache**: Investigating whether compression techniques can reduce the memory footprint of KV caches during inference.
