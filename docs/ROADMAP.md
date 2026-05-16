# Roadmap

## Known Limitations (Current)

These are **real, documented limitations** of the current KMC implementation. They are not bugs — they are scope boundaries:

1. **KMC does NOT reduce VRAM during inference.** Compressed archives must be fully unpacked before a model can be loaded. Block-loading is future research.
2. **KMC does NOT modify model weights.** Compression is strictly lossless. No quality degradation occurs at any point.
3. **Block-loading is not implemented.** The manifest contains per-block offsets, but on-demand decompression of individual blocks is a future feature.
4. **GGUF block-aware compression is future work.** GGUF files are detected and their headers are parsed, but format-specific compression strategies (e.g., skipping already-quantized blocks) are not yet implemented.
5. **No fixed compression ratios.** Results depend heavily on model format, data type, and content. Synthetic benchmarks produce misleadingly high ratios and should not be used as references.
6. **KMC is not quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses files for storage and transfer.

## Phase 1: Foundation (v0.1.0) — Completed

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

## Phase 2: Hardening (v0.2.0) — Current

Focus on security, robustness, and realistic testing:

- [x] Harden `unpack()` with `safe_join_extract_path()` function
- [x] Comprehensive path traversal tests (null bytes, control chars, symlinks, absolute paths, `..` components)
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
- [x] Kimari CLI integration adapters (`src/kmc/integrations/kimari.py`)
- [x] Documented limitations in README and ROADMAP
- [ ] Add decompressed size limits to prevent zip bombs
- [ ] Add property-based testing with Hypothesis
- [ ] Test with corrupted archives (fuzzing)
- [ ] Create real benchmarks with small models (e.g., GPT-2 from Hugging Face)
- [ ] Add `--verbose` and `--quiet` flags to CLI
- [ ] Support stdin/stdout for pipe-friendly workflows
- [ ] Add progress bars for large archives

## Phase 3: Performance (v0.3.0)

Optimize throughput and resource usage:

- [ ] Parallel block compression/decompression with `concurrent.futures`
- [ ] Memory-mapped archive reading for large files
- [ ] Streaming pack/unpack for minimal memory footprint
- [ ] Benchmark against `tar.zst`, `zip`, and ZipNN
- [ ] Profile and optimize hot paths
- [ ] Add compression level presets (fast/balanced/max)

## Phase 4: Format-Aware Compression (v0.4.0)

Leverage knowledge of AI model formats for better compression:

- [ ] Real safetensors support: align block boundaries with tensor boundaries
- [ ] Skip already-compressed quantized blocks in GGUF
- [ ] Delta compression for LoRA adapters relative to base models
- [ ] Weight sharing detection across model files
- [ ] Format-specific compression hints in manifest
- [ ] Benchmark against ZipNN

## Phase 5: Kimari Integration (v0.5.0)

Integration with the Kimari ecosystem:

- [ ] `kimari compress` command that wraps KMC
- [ ] KimariDB storage backend integration
- [ ] Archive cataloging and metadata search
- [ ] Integration with Hugging Face Hub for download-cache-verify workflow
- [ ] Registry of pre-compressed model archives
- [ ] Documentation for Hugging Face integration

## Phase 6: Advanced Features (v1.0.0)

- [ ] Block-level loading (partial decompression on demand)
- [ ] Block server for remote block fetching
- [ ] Checkpoint/gradients compression for distributed training
- [ ] Encrypted archives for sensitive models
- [ ] Archive signing and verification
- [ ] Multi-part archives for very large models
- [ ] Python API for programmatic access (beyond CLI)
- [ ] C/Rust extension for core compression operations
- [ ] Minimum GGUF parser for block-level access

## Research Directions

These are exploratory areas that may or may not be incorporated:

- **Neural compression**: Using small neural networks to compress weight matrices more efficiently than general-purpose codecs.
- **Quantization-aware compression**: Exploiting the structure of quantized weights (e.g., GGUF quantization levels) for better compression.
- **Distributed compression**: Coordinating compression across multiple nodes for federated learning scenarios.
- **Incremental archives**: Supporting efficient updates to existing archives without full repack.
- **Compressed KV cache**: Investigating whether compression techniques can reduce the memory footprint of KV caches during inference (distinct from model file compression).
