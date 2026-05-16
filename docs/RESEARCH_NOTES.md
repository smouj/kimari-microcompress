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
2. **Tensor-type-aware compression**: Use dtype information from safetensors headers to apply dtype-specific compression (e.g., XOR delta encoding for float16 weights). — *Partially addressed in v0.4 with BytePlane and FloatPlane codecs.*
3. **Cross-file deduplication**: Detect identical tensors across different files in the same archive and store them once.
4. **Sparse tensor support**: Compress sparse tensors by storing only non-zero values with index information.
5. **Neural compression**: Train small neural networks to compress weight matrices more efficiently than general-purpose codecs.

---

## BytePlane Codec Design Rationale (v0.4)

### The Core Observation

Floating-point numbers in AI model weights are stored in multi-byte formats (BF16 = 2 bytes, FP16 = 2 bytes, FP32 = 4 bytes). When stored sequentially in memory, the bytes from different numerical positions are interleaved:

```
BF16: [sign_exp_0, mantissa_0, sign_exp_1, mantissa_1, ...]
FP32: [byte0_a, byte1_a, byte2_a, byte3_a, byte0_b, byte1_b, ...]
```

General-purpose compressors (zstd, zlib) treat this as an opaque byte stream. However, bytes at the same position within each floating-point element tend to have very different statistical properties:

- **Sign/exponent bytes** (high bytes in BF16/FP16, first two bytes in FP32) have low entropy because exponents cluster in a narrow range for trained weights.
- **Mantissa bytes** (low bytes in BF16/FP16, last two bytes in FP32) have higher entropy but still benefit from being grouped together.

### Byte-Plane Separation

BytePlane separates bytes by their position within each element:

```
Input:  [a0, b0, a1, b1, a2, b2, ...]  (element_size=2)
Plane0: [a0, a1, a2, ...]               (all high bytes)
Plane1: [b0, b1, b2, ...]               (all low bytes)
```

After separation, each plane is a contiguous stream of bytes from the same numerical position. These streams have much more uniform statistics and compress better with standard codecs like zstd.

### Tradeoffs

**Advantages:**
- Simple transformation with negligible computational overhead.
- Preserves all information (fully lossless).
- Works for any fixed-width numeric type, not just floats.
- Significantly improves compressibility for BF16/FP16/FP32 data.

**Disadvantages:**
- Requires knowing the element size (inferred from dtype context or guessed from data alignment).
- Misaligned tail bytes (when data length is not divisible by element_size) are handled separately, adding minor complexity.
- For non-numeric data or already-compressed data, byte-plane separation may hurt rather than help. The automatic selector handles this by falling back to zstd/raw.
- The transformation adds one level of indirection: decompression requires metadata (element_size, inner_codec) stored in the manifest.

### Design Decisions

1. **Plane concatenation vs. separate compression**: BytePlane concatenates all planes before compressing with the inner codec. An alternative would be to compress each plane separately. Concatenation was chosen because: (a) it's simpler, (b) zstd can exploit cross-plane patterns, and (c) separate compression would add more metadata overhead for small blocks.

2. **Inner codec selection**: BytePlane uses zstd (level 3) if available, otherwise zlib. No raw fallback at the BytePlane level because if BytePlane + zstd doesn't help, the automatic selector will choose plain zstd instead.

3. **No bit-level separation**: BytePlane operates at the byte level, not the bit level. Bit-level separation (sign/exponent/mantissa) is handled by the FloatPlane codec, which is a separate, more specialized codec.

---

## FloatPlane Codec Design Rationale (v0.4)

### The Core Observation

BytePlane separates bytes by position, but within a single byte, there may still be mixed bit fields. For example, in BF16 (1 sign bit, 8 exponent bits, 7 mantissa bits), the high byte contains the sign bit and the top 7 exponent bits. The low byte contains the bottom exponent bit and all 7 mantissa bits.

FloatPlane goes further by separating at the bit-field level:

```
BF16 value: [S][EEEEEEEE][MMMMMMM]
             sign  exponent  mantissa
```

Each component is extracted, packed efficiently, and compressed independently.

### Why Separate Sign, Exponent, and Mantissa?

1. **Sign bits**: In trained neural network weights, the vast majority of values are positive (or have a consistent sign pattern within a layer). A stream of mostly-0 sign bits compresses to nearly nothing.

2. **Exponent values**: Weight magnitudes tend to cluster around a small number of values. In BF16 (8 exponent bits, range 0-255), most weights in a given layer might use only 10-20 distinct exponent values. This creates a highly compressible stream.

3. **Mantissa values**: While mantissa bits have the highest entropy, they still benefit from being separated from the sign and exponent. Adjacent mantissa values may share prefixes or patterns that zstd can exploit.

### Bit-Level Operations

FloatPlane operates entirely on integer bit patterns. No conversion to/from Python `float` occurs. The process is:

1. Read each element as an unsigned integer (uint16 for BF16/FP16, uint32 for FP32).
2. Extract sign, exponent, and mantissa using bit masks.
3. Pack sign bits (8 per byte), exponents (minimal bytes per value), and mantissas (minimal bytes per value) into separate byte streams.
4. Compress each stream independently with zstd or zlib.

This approach is inherently lossless: the original bit pattern is reconstructed exactly by OR-ing the components back together.

### Supported Dtypes

| Dtype | Total Bits | Sign | Exponent | Mantissa | Byte Size |
|-------|-----------|------|----------|----------|-----------|
| BF16  | 16        | 1    | 8        | 7        | 2         |
| FP16  | 16        | 1    | 5        | 10       | 2         |
| FP32  | 32        | 1    | 8        | 23       | 4         |

Unsupported dtypes (INT8, INT4, etc.) cause FloatPlane to fall back to BytePlane internally.

### Tradeoffs

**Advantages:**
- Most granular floating-point decomposition available in KMC.
- Sign plane compresses extremely well for typical neural network weights.
- Exponent plane benefits from value clustering.
- Independent plane compression allows each to use the optimal strategy.

**Disadvantages:**
- Highest computational cost of all KMC codecs (3 separate compress operations per block, plus bit manipulation overhead).
- Requires dtype context (cannot operate without knowing the float format).
- For small blocks or blocks with few elements, the overhead of separate plane compression may exceed the benefit.
- Fallback to BytePlane when dtype is unknown or unsupported adds complexity.

### Design Decisions

1. **Independent vs. joint plane compression**: FloatPlane compresses each plane (sign, exponent, mantissa) independently. Joint compression could exploit cross-plane correlations but would add complexity and reduce the entropy separation benefit.

2. **Payload format**: Each plane's compressed data is length-prefixed, allowing the decompressor to locate each plane without metadata. The format is: `[sign_len(4)][sign_data][exp_len(4)][exp_data][mantissa_len(4)][mantissa_data][tail_len(4)][tail]`.

3. **Fallback chain**: When FloatPlane cannot operate (no dtype, unsupported dtype), it falls back to BytePlane and records `"transform": "byteplane_fallback"` in the metadata. This ensures the codec always produces valid output, even in degraded conditions.

4. **No attempt to compress already-quantized data**: FloatPlane only operates on BF16, FP16, and FP32. Quantized formats (INT8, INT4, GGUF quantization levels) do not have a standard floating-point bit layout and are handled by the standard zstd/zlib pipeline.

---

## Future Codec Research Directions

1. **XOR delta encoding for weight matrices**: Instead of compressing raw weight bytes, XOR each row with the previous row. If adjacent rows are similar (common in trained models), the XOR result will be mostly zeros, which compresses extremely well. This could be particularly effective for large weight matrices.

2. **Per-layer codec tuning**: Different layers in a transformer have different weight distributions. Attention layers may benefit from FloatPlane, while embedding layers may benefit from BytePlane. The automatic selector could be extended to consider layer type, not just dtype.

3. **Dictionary-trained zstd**: Training a zstd dictionary on representative model blocks could improve compression ratio for the inner codec step of BytePlane/FloatPlane. This would require a two-pass approach: first pass to collect training data, second pass to compress with the dictionary.

4. **SIMD-accelerated plane separation**: Byte-plane and float-plane separation are embarrassingly parallel operations. Using NumPy or a C extension for plane separation could significantly speed up the transformation step.

5. **Adaptive plane selection**: Instead of always separating into sign/exponent/mantissa, an adaptive codec could measure the entropy of different plane configurations and choose the one that minimizes total compressed size. For example, for BF16 with uniform exponents, separating into just 2 planes (sign+exponent, mantissa) might be better than 3 planes.

6. **GGUF quantization-aware compression**: Understanding the internal structure of GGUF quantization blocks (Q4_K, Q5_1, etc.) could enable codecs that skip already-quantized blocks and focus on compressible metadata and vocabulary data.
