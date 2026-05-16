# Roadmap

## Known Limitations (Current)

These are **real, documented limitations** of the current KMC implementation. They are not bugs — they are scope boundaries:

1. **KMC does NOT reduce VRAM during inference.** Compressed archives must be fully unpacked before a model can be loaded. Block-loading is future research.
2. **KMC does NOT modify model weights.** Compression is strictly lossless. No quality degradation occurs at any point.
3. **Block-loading is not implemented.** The manifest contains per-block offsets, but on-demand decompression of individual blocks is a future feature.
4. **GGUF block-aware compression is future work.** GGUF files are detected and their headers are parsed, but format-specific compression strategies (e.g., skipping already-quantized blocks) are not yet implemented.
5. **No fixed compression ratios.** Results depend heavily on model format, data type, and content. Synthetic benchmarks produce misleadingly high ratios and should not be used as references.
6. **KMC is not quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses files for storage and transfer.
7. **Tensor-aware mode is structural, not algorithmic.** The `--tensor-aware` flag aligns block boundaries to tensor boundaries but does not yet implement tensor-specific codecs (e.g., sign/exponent/mantissa separation for BF16/FP16). That is planned for v0.4.
8. **GGUF tensor metadata is not yet parsed.** The GGUF parser reads the header (magic, version, tensor count, KV count) but does not parse individual tensor descriptors or metadata values. Full parsing is planned for v0.5.

## v0.1 — MVP Archive

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

## v0.2 — Security + Verification + Benchmark

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

## v0.3 — Safetensors + ZipNN Comparison + GGUF Parser (Current)

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

## v0.4 — Tensor-Aware Block Codec Experiments

Real tensor-specific compression algorithms:

- [ ] BF16/FP16 sign/exponent/mantissa separation codec
- [ ] Per-dtype compression strategies
- [ ] Block-level codec selection based on tensor dtype
- [ ] Benchmark tensor-aware codecs against generic compression
- [ ] Tests with small real models (GPT-2, BERT-base)
- [ ] Public reproducible comparison against ZipNN

## v0.5 — LoRA/Checkpoint Compression Workflows

Specialized compression for adapters and training artifacts:

- [ ] Delta compression for LoRA adapters relative to base models
- [ ] Checkpoint/gradients compression
- [ ] GGUF tensor metadata parsing
- [ ] GGUF block-aware compression (skip already-quantized blocks)
- [ ] Optimizer state compression for training checkpoints
- [ ] Weight sharing detection across model files

## v0.6 — Kimari CLI Integration

Full integration with the Kimari ecosystem:

- [ ] `kimari compress` command in Kimari CLI
- [ ] `kimari decompress` command
- [ ] `kimari verify-compress` command
- [ ] `kimari bench-compress` command
- [ ] Shared configuration (block size, compression level)
- [ ] Progress reporting integration
- [ ] KimariDB storage backend integration

## v0.7 — Experimental Loader/Runtime Work

Block-level loading and runtime integration:

- [ ] Block-level loading (partial decompression on demand)
- [ ] Block server for remote block fetching
- [ ] Memory-mapped archive reading for large files
- [ ] Streaming pack/unpack for minimal memory footprint
- [ ] Python API for programmatic block access
- [ ] Integration with model loading frameworks

## Future Research Directions

These are exploratory areas that may or may not be incorporated:

- **Neural compression**: Using small neural networks to compress weight matrices more efficiently than general-purpose codecs.
- **Quantization-aware compression**: Exploiting the structure of quantized weights (e.g., GGUF quantization levels) for better compression.
- **Distributed compression**: Coordinating compression across multiple nodes for federated learning scenarios.
- **Incremental archives**: Supporting efficient updates to existing archives without full repack.
- **Compressed KV cache**: Investigating whether compression techniques can reduce the memory footprint of KV caches during inference.
