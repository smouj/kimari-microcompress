# Research Notes

## Lossless Compression for AI Models

### ZipNN (IBM Research)

**Paper**: "A Lossless Compression for AI Models" — IBM Research  
**URL**: https://research.ibm.com/publications/a-lossless-compression-for-ai-models

Key findings:
- ZipNN achieves approximately 1/3 size reduction on popular AI models without any modification to the weights.
- In some cases, compression exceeds 50% reduction.
- The approach is lossless — weights are byte-identical after decompression.
- This validates the core premise of KMC: AI model files contain significant compressible redundancy that general-purpose tools don't fully exploit.

Implications for KMC:
- Per-block compression with zstd should achieve similar ratios for safetensors files.
- ZipNN may use format-specific tricks that KMC can incorporate as codec improvements.
- The "no weight modification" principle aligns with KMC's design.

### NetZIP (IBM Research)

**Paper**: "NetZIP: Algorithm-Hardware Co-Design of In-Network Lossless Compression for Distributed Large Model Training" — IBM Research  
**URL**: https://research.ibm.com/publications/netzip-algorithmhardware-co-design-of-in-network-lossless-compression-for-distributed-large-model-training

Key findings:
- Lossless compression can be applied to gradients and activations during distributed training.
- Network bandwidth is a bottleneck in distributed training, and compression helps.
- Hardware-aware compression design achieves high throughput.

Implications for KMC:
- Checkpoint/gradients compression is a viable future direction.
- Training checkpoints (optimizer states, gradients) may have different compression characteristics than inference weights.
- Block-oriented design maps naturally to gradient checkpoint blocks.

## AI Model Formats

### safetensors (Hugging Face)

**Documentation**: https://huggingface.co/docs/safetensors/index

Key characteristics:
- Designed as a secure alternative to pickle-based formats.
- Uses memory-mapped I/O for fast loading without copying.
- Header is JSON, followed by tensor data in a flat layout.
- No arbitrary code execution (unlike pickle).
- Supported by Hugging Face Transformers, Diffusers, and growing ecosystem.

Why KMC prioritizes safetensors:
- Growing adoption as the standard format for model distribution.
- Memory-mapped design aligns with KMC's block-oriented architecture.
- JSON header is compressible and contains structural information.
- Tensor data (float32/float16) tends to compress well with zstd.

### GGUF (llama.cpp)

**Documentation**: https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format

Key characteristics:
- Binary format with a header containing metadata key-value pairs.
- Supports multiple quantization levels (Q4_0, Q5_1, Q8_0, etc.).
- Designed for efficient CPU inference via llama.cpp.
- Single-file distribution model.

Why GGUF is a future integration target:
- Already quantized, so additional compression has limited benefit.
- However, vocabulary data, metadata, and padding can still be compressed.
- GGUF's quantization is lossy (modifying weights); KMC's compression is lossless (preserving weights).
- Block-loading of GGUF files could enable partial model loading.

## Compression Algorithms

### Zstandard (zstd)

- Developed by Facebook (Meta).
- Fast decompression (10x faster than zlib at comparable ratios).
- Supports dictionary compression for many small blocks.
- Wide range of compression levels (1-22).
- Available as a Python package (`zstandard`).

KMC uses zstd as the primary codec because:
- Best speed/ratio tradeoff for AI model data.
- Dictionary mode could improve compression across similar blocks.
- Well-maintained and widely available.

### zlib / DEFLATE

- Ubiquitous compression algorithm.
- Built into Python's standard library.
- Moderate compression ratio and speed.
- Used as KMC's fallback when zstd is not available.

### Raw (uncompressed)

- No compression applied.
- Used when compression doesn't reduce size.
- Ensures KMC never expands data.

## Future Research Directions

1. **Dictionary compression**: Train a zstd dictionary on a set of model blocks to improve per-block compression ratios.
2. **Tensor-type-aware compression**: Use dtype information from safetensors headers to apply dtype-specific compression (e.g., XOR delta encoding for float16 weights).
3. **Cross-file deduplication**: Detect identical tensors across different files in the same archive and store them once.
4. **Sparse tensor support**: Compress sparse tensors by storing only non-zero values with index information.
5. **Neural compression**: Train small neural networks to compress weight matrices more efficiently than general-purpose codecs.
