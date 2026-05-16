# Roadmap

## Phase 1: Foundation (Current — v0.1.0)

The initial release establishes the core infrastructure for lossless AI model compression:

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

## Phase 2: Hardening (Next — v0.2.0)

Focus on security, robustness, and real-world testing:

- [ ] Harden `unpack()` with comprehensive path traversal tests
- [ ] Add manifest size limits to prevent DoS
- [ ] Add decompressed size limits to prevent zip bombs
- [ ] Create real benchmarks with small models (e.g., GPT-2, small LLaMA)
- [ ] Add property-based testing with Hypothesis
- [ ] Test with corrupted archives (fuzzing)
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

- [ ] Align block boundaries with tensor boundaries in safetensors
- [ ] Skip already-compressed quantized blocks in GGUF
- [ ] Delta compression for LoRA adapters relative to base models
- [ ] Weight sharing detection across model files
- [ ] Format-specific compression hints in manifest

## Phase 5: Kimari Integration (v0.5.0)

Integration with the Kimari ecosystem:

- [ ] `kimari compress` command that wraps KMC
- [ ] KimariDB storage backend integration
- [ ] Archive cataloging and metadata search
- [ ] Integration with Hugging Face Hub for download-cache-verify workflow
- [ ] Registry of pre-compressed model archives

## Phase 6: Advanced Features (v1.0.0)

- [ ] Block-level loading (partial decompression on demand)
- [ ] Block server for remote block fetching
- [ ] Checkpoint/gradients compression for distributed training
- [ ] Encrypted archives for sensitive models
- [ ] Archive signing and verification
- [ ] Multi-part archives for very large models
- [ ] Python API for programmatic access (beyond CLI)
- [ ] C/Rust extension for core compression operations

## Research Directions

These are exploratory areas that may or may not be incorporated:

- **Neural compression**: Using small neural networks to compress weight matrices more efficiently than general-purpose codecs.
- **Quantization-aware compression**: Exploiting the structure of quantized weights (e.g., GGUF quantization levels) for better compression.
- **Distributed compression**: Coordinating compression across multiple nodes for federated learning scenarios.
- **Incremental archives**: Supporting efficient updates to existing archives without full repack.
