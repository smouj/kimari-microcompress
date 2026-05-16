# Kimari MicroCompress (KMC)

**Reversible lossless compression for AI models: safetensors, GGUF, checkpoints, LoRA/QLoRA and future block-loading.**

[![CI](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Important Limitations

**Please read before using KMC:**

- **KMC does NOT reduce VRAM during inference.** It is designed for storage, transfer, and verification -- not runtime memory optimization.
- **KMC does NOT modify model weights.** Compression is lossless and reversible; every byte is preserved exactly.
- **Block-loading (partial decompression) is future research.** The current format stores blocks with offsets, but on-demand block serving is not yet implemented.
- **GGUF-aware compression is experimental.** The `--gguf-aware` flag adjusts codec selection for quantized GGUF tensors but does not yet implement block-level GGUF-specific compression strategies.
- **No fixed compression ratios should be assumed without real benchmarks.** Results vary significantly by model format, data type, and content. Synthetic benchmarks do not represent real-world ratios.
- **KMC is not a replacement for quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses the already-quantized files for storage/transfer.
- **No pickle is used.** KMC never deserializes pickle-based files (optimizer.pt, training_args.bin, pytorch_model.bin). It only records their presence, size, and hash.
- **KMC is lossless only.** There is no lossy mode and no weight modification of any kind.

## KMC v0.5.0-alpha Focus

This release introduces GGUF tensor metadata parsing, specialized workflows for LoRA adapters and training checkpoints, experimental GGUF-aware compression, and real small-model benchmarks:

- **GGUF tensor metadata parser** -- Full tensor info extraction (name, shape, type, offset, estimated size) and quantization summary (e.g., Q4_K: 201, F32: 1)
- **Experimental `--gguf-aware` compression mode** -- Adjusts codec selection for quantized GGUF tensors, avoiding float-aware transforms on already-quantized data; records GGUF format metadata in the manifest
- **LoRA/PEFT adapter workflows** -- `kmc pack-lora` and `kmc inspect --lora` for dedicated LoRA adapter handling with rank, target modules, and base model metadata
- **Training checkpoint workflows** -- `kmc pack-checkpoint` and `kmc inspect --checkpoint` for training checkpoint directories, with step detection and component identification
- **Manifest v4** -- Adds `artifact_type`, `artifact_metadata`, and `format_metadata` fields for structured artifact and format information
- **Artifact auto-detection** -- Automatic classification as `huggingface_model`, `gguf_model`, `lora_adapter`, `training_checkpoint`, or `unknown`
- **Real small-model benchmark script** -- `scripts/bench_small_hf_model.py` for reproducible benchmarking on actual HuggingFace models
- **Kimari CLI integration preparation** -- New adapter wrappers for `compress-lora`, `compress-checkpoint`, and `inspect-model`
- **228 tests passing**

> **Important disclaimer:** KMC v0.5 still does not reduce inference VRAM. It compresses model storage and transfer artifacts. The `--gguf-aware` flag is experimental and may change in future releases.

## Overview

Kimari MicroCompress (KMC) is an experimental tool for **lossless, reversible compression** of AI model files. It focuses on **storage, transfer, verification, and packaging** without modifying the original weights. The approach is grounded in the observation that AI model files -- particularly `safetensors` and quantized formats -- contain significant redundancy that general-purpose compression tools don't exploit optimally.

**Key principle:** Every byte that goes in must come out identically. KMC provides byte-exact roundtrip integrity verified via SHA-256 hashes at both the file and block level.

## Features

| Feature | Status |
|---------|--------|
| `kmc pack` -- Compress files/directories | Working |
| `kmc pack --tensor-aware` -- Tensor-aware block alignment | Working |
| `kmc pack --gguf-aware` -- Experimental GGUF-aware compression | Experimental |
| `kmc pack-lora` -- LoRA adapter workflow | Working |
| `kmc pack-checkpoint` -- Training checkpoint workflow | Working |
| `kmc unpack` -- Decompress archives (path-safe) | Working |
| `kmc verify` -- Full verification report | Working |
| `kmc inspect` -- AI model inspection with tensor metadata | Working |
| `kmc inspect --json` -- JSON output for scripting | Working |
| `kmc inspect --tensors` -- Detailed tensor information | Working |
| `kmc inspect --lora` -- LoRA adapter inspection | Working |
| `kmc inspect --checkpoint` -- Training checkpoint inspection | Working |
| `kmc inspect --gguf` -- GGUF model inspection with tensor details | Working |
| `kmc bench` -- Benchmark with codec comparison | Working |
| `kmc bench --compare-codecs` -- Multi-codec comparison | Working |
| `kmc bench --compare-zipnn` -- ZipNN comparison | Working |
| `kmc pack/bench --codec auto\|byteplane\|floatplane\|zstd\|zlib\|raw` | Working |
| `kmc inspect --compression` -- Codec usage summary | Working |
| `.kmc` archive format with JSON manifest (v4) | Working |
| zstd / zlib / raw / byteplane / floatplane codec selection | Working |
| Automatic codec selector (dtype-based) | Working |
| GGUF tensor metadata parser (names, shapes, types, offsets, sizes) | Working |
| GGUF quantization summary (Q4_K, Q5_1, F32, etc.) | Working |
| SHA-256 per-file and per-block hashing | Working |
| 256 KiB micro-blocks (configurable) | Working |
| AI format detection (safetensors, GGUF, LoRA, shards, etc.) | Working |
| Artifact auto-detection (huggingface_model, gguf_model, lora_adapter, training_checkpoint) | Working |
| safetensors real tensor metadata (names, shapes, dtypes, offsets) | Working |
| safetensors shard detection | Working |
| LoRA/PEFT adapter detection with rank and target modules | Working |
| GGUF header parsing (version, endianness, tensor count) | Working |
| Path traversal protection in unpack | Working |
| Manifest validation (duplicates, codecs, sizes) | Working |
| Full verification report with block/file hash checks | Working |
| Benchmark JSON export with environment metadata | Working |
| Kimari CLI integration adapters (compress, compress-lora, compress-checkpoint) | Working |
| `safetensors` optional dependency support | Working |
| `zipnn` optional dependency support | Working |
| Real model benchmarks (bench_small_hf_model.py) | Working |
| BytePlane codec (byte-plane separation) | Working |
| FloatPlane codec (sign/exp/mantissa separation) | Working |
| Backward compatible with .kmc v0.2/v0.3/v0.4 | Working |
| GGUF block-level compression | Research |
| Block-loading (partial decompression) | Research |

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/smouj/kimari-microcompress.git
cd kimari-microcompress
pip install -e ".[dev]"

# With safetensors optional dependency (better header parsing)
pip install -e ".[safetensors]"

# With ZipNN optional dependency (for benchmark comparison)
pip install -e ".[zipnn]"

# All optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- `zstandard` (required, for best compression)
- `zlib` (built-in, used as fallback)
- `safetensors` (optional, for enhanced header parsing)
- `zipnn` (optional, for benchmark comparison)

## Quick Start

```bash
# Pack a model directory
kmc pack ./my-model ./my-model.kmc

# Pack with tensor-aware mode (recommended for safetensors)
kmc pack ./my-model ./my-model.kmc --tensor-aware

# Pack with GGUF-aware mode (experimental, for GGUF files)
kmc pack ./my-model ./my-model.kmc --gguf-aware

# Pack with a specific codec
kmc pack ./my-model ./my-model.kmc --codec byteplane
kmc pack ./my-model ./my-model.kmc --codec floatplane

# Pack a LoRA adapter
kmc pack-lora ./my-lora-adapter ./my-lora.kmc

# Pack a training checkpoint
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc

# Verify integrity (full report)
kmc verify ./my-model.kmc

# Inspect archive manifest
kmc inspect ./my-model.kmc

# Inspect AI model directory (detects formats, reads tensor metadata)
kmc inspect ./my-model/ --tensors

# Inspect as LoRA adapter
kmc inspect ./my-lora/ --lora

# Inspect as training checkpoint
kmc inspect ./checkpoint-1000/ --checkpoint

# Inspect GGUF file with tensor details
kmc inspect ./model.gguf --gguf

# Inspect with JSON output
kmc inspect ./my-model/ --json

# Unpack to a directory
kmc unpack ./my-model.kmc ./restored-model/

# Run benchmark with JSON output
kmc bench ./my-model ./my-model-bench.kmc --json --output report.json

# Benchmark with codec comparison (tries all codecs)
kmc bench ./my-model ./my-model-bench.kmc --compare-codecs

# Benchmark with a specific codec
kmc bench ./my-model ./my-model-bench.kmc --codec floatplane

# Benchmark with ZipNN comparison
kmc bench ./my-model ./my-model-bench.kmc --compare-zipnn
```

## KMC Archive Format

The `.kmc` format is designed for verifiable, block-oriented storage:

```
+-------------------------------------+
| Magic: "KMC\x00\x01\x00\x00\x00"   |  8 bytes
+-------------------------------------+
| Manifest length: uint64 BE          |  8 bytes
+-------------------------------------+
| Manifest: JSON (UTF-8 encoded)     |  Variable
|  - version, tool info              |
|  - file entries with paths & hashes|
|  - block entries with codecs       |
|  - per-block codec_metadata (v3+)  |
|  - tensor entries (v2+, optional)  |
|  - artifact_type (v4+)             |
|  - artifact_metadata (v4+)         |
|  - format_metadata (v4+)           |
|  - total sizes and ratios          |
+-------------------------------------+
| Block data: concatenated           |  Variable
|  - Each block independently         |
|    compressed (zstd/zlib/raw)      |
|  - SHA-256 verified per block      |
+-------------------------------------+
```

See [FORMAT_SPEC.md](docs/FORMAT_SPEC.md) for the complete specification.

## Security

KMC takes extraction security seriously:

- **Path traversal protection**: All file paths in the manifest are validated before extraction. Paths with `..`, absolute paths, null bytes, and control characters are rejected.
- **Symlink protection**: KMC refuses to overwrite existing symlinks during unpack.
- **Duplicate path detection**: Manifests with duplicate file paths are rejected.
- **Manifest size limits**: Oversized manifests are rejected to prevent denial-of-service attacks.
- **Block hash verification**: Every block is verified against its SHA-256 hash before decompression.
- **File hash verification**: Reconstructed files are verified against their SHA-256 hash.
- **No pickle deserialization**: KMC never uses pickle to inspect or load model data. Pickle-based files (optimizer.pt, training_args.bin, pytorch_model.bin) are detected and compressed as raw bytes only.

See [SECURITY_MODEL.md](docs/SECURITY_MODEL.md) for the complete security model.

## Architecture

```
src/kmc/
+-- archive.py          # Core pack/unpack/verify with security checks
+-- benchmark.py        # Performance benchmarking with codec + ZipNN comparison
+-- cli.py              # Command-line interface
+-- codecs/             # Compression codec subpackage (v0.4+)
|   +-- __init__.py     # Public codec API
|   +-- base.py         # Codec protocol, CodecContext, CodecResult
|   +-- byteplane.py    # BytePlane codec (byte-plane separation)
|   +-- floatplane.py   # FloatPlane codec (sign/exp/mantissa separation)
|   +-- registry.py     # Codec registry (discover/instantiate by name)
|   +-- selector.py     # Automatic codec selector (dtype-based candidates)
|   +-- legacy.py       # Legacy CodecId/compress_block API (v0.2/v0.3 compat)
|   +-- raw.py          # Raw passthrough codec
|   +-- zlib_codec.py   # zlib codec
|   +-- zstd_codec.py   # zstd codec
+-- codecs.py           # Legacy codec module (see codecs/ subpackage)
+-- formats/
|   +-- __init__.py     # Format module registry
|   +-- safetensors.py  # Safetensors metadata, shards, LoRA detection
|   +-- gguf.py         # GGUF header + tensor metadata parsing (v0.5+)
+-- workflows/           # Artifact-specific workflows (v0.5+)
|   +-- __init__.py     # Workflow module registry
|   +-- lora.py         # LoRA/PEFT adapter detection and packing
|   +-- checkpoint.py   # Training checkpoint detection and packing
+-- hashing.py          # SHA-256 integrity hashing
+-- inspector.py        # AI model format detection with metadata
+-- manifest.py         # KMC manifest (v4: artifact_type, format_metadata)
+-- tensor_inspector.py # Legacy safetensors metadata (see formats/safetensors.py)
+-- gguf.py             # Legacy GGUF module (see formats/gguf.py)
+-- integrations/
    +-- kimari.py       # Kimari CLI integration adapters
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Technical Foundation

KMC's approach is informed by research and industry practice:

- **ZipNN** (IBM Research) demonstrates that lossless compression specific to AI models can save ~1/3 of size on popular models, and >50% in some cases, without changing weights.
- **safetensors** (Hugging Face) is treated as the priority format because it's secure, fast, and avoids `pickle` vulnerabilities.
- **GGUF** (llama.cpp) is the standard binary format for quantized models. KMC v0.5 adds full tensor metadata parsing and experimental GGUF-aware compression.
- **NetZIP** (IBM Research) explores lossless compression for gradients and activations in distributed training -- a research direction documented for KMC's roadmap.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- Design decisions and module structure
- [Format Specification](docs/FORMAT_SPEC.md) -- Complete `.kmc` format spec (v4)
- [Security Model](docs/SECURITY_MODEL.md) -- Threat model and mitigations
- [Roadmap](docs/ROADMAP.md) -- Development priorities
- [Benchmark Plan](docs/BENCHMARK_PLAN.md) -- Performance testing strategy
- [Research Notes](docs/RESEARCH_NOTES.md) -- Technical references and codec design rationale
- [Real Model Benchmark](docs/REAL_MODEL_BENCHMARK.md) -- Running benchmarks with HuggingFace models
- [Kimari Integration](docs/KIMARI_INTEGRATION.md) -- Integration with Kimari CLI
- [Hugging Face Workflow](docs/HUGGINGFACE_WORKFLOW.md) -- Working with Hugging Face models
- [GGUF Support](docs/GGUF_SUPPORT.md) -- GGUF parsing and `--gguf-aware` mode
- [LoRA Workflow](docs/LORA_WORKFLOW.md) -- LoRA adapter compression and inspection
- [Checkpoint Workflow](docs/CHECKPOINT_WORKFLOW.md) -- Training checkpoint compression and inspection

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -q

# Lint
ruff check .

# Format check
ruff format --check .

# CLI help
kmc --help

# Create demo model and test
python scripts/create_demo_model.py
bash scripts/run_demo.sh
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
