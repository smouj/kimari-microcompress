# Kimari MicroCompress (KMC)

**Reversible lossless compression for AI models: safetensors, GGUF, checkpoints, LoRA/QLoRA and future block-loading.**

[![CI](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚠️ Important Limitations

**Please read before using KMC:**

- **KMC does NOT reduce VRAM during inference.** It is designed for storage, transfer, and verification — not runtime memory optimization.
- **KMC does NOT modify model weights.** Compression is lossless and reversible; every byte is preserved exactly.
- **Block-loading (partial decompression) is future research.** The current format stores blocks with offsets, but on-demand block serving is not yet implemented.
- **GGUF block-aware compression is future work.** GGUF files are detected and parsed, but format-specific compression strategies are planned for a later phase.
- **No fixed compression ratios should be assumed without real benchmarks.** Results vary significantly by model format, data type, and content. Synthetic benchmarks do not represent real-world ratios.
- **KMC is not a replacement for quantization.** If you need smaller models for inference, use quantization (GGUF Q4_K, GPTQ, AWQ, etc.). KMC is complementary: it compresses the already-quantized files for storage/transfer.

## KMC v0.3.0-alpha Focus

This release focuses on real safetensors support and professional tooling:

- **Real safetensors metadata inspection** — Read tensor names, dtypes, shapes, and byte offsets without loading weights
- **Tensor-aware manifest entries** — Block boundaries aligned to tensor boundaries with `--tensor-aware`
- **Optional ZipNN benchmark comparison** — Measure against ZipNN with fair, reproducible results
- **Minimal GGUF parser** — Header parsing with version, endianness, and tensor count detection
- **Kimari CLI adapter preparation** — Clean integration layer for `kimari compress/decompress/verify-compress/bench-compress`
- **Hugging Face workflow documentation** — Complete guide for compressing models from Hugging Face Hub

## Overview

Kimari MicroCompress (KMC) is an experimental tool for **lossless, reversible compression** of AI model files. It focuses on **storage, transfer, verification, and packaging** without modifying the original weights. The approach is grounded in the observation that AI model files — particularly `safetensors` and quantized formats — contain significant redundancy that general-purpose compression tools don't exploit optimally.

**Key principle:** Every byte that goes in must come out identically. KMC provides byte-exact roundtrip integrity verified via SHA-256 hashes at both the file and block level.

## Features

| Feature | Status |
|---------|--------|
| `kmc pack` — Compress files/directories | ✅ Working |
| `kmc pack --tensor-aware` — Tensor-aware block alignment | ✅ Working |
| `kmc unpack` — Decompress archives (path-safe) | ✅ Working |
| `kmc verify` — Full verification report | ✅ Working |
| `kmc inspect` — AI model inspection with tensor metadata | ✅ Working |
| `kmc inspect --json` — JSON output for scripting | ✅ Working |
| `kmc inspect --tensors` — Detailed tensor information | ✅ Working |
| `kmc bench` — Benchmark with codec comparison | ✅ Working |
| `kmc bench --compare-zipnn` — ZipNN comparison | ✅ Working |
| `.kmc` archive format with JSON manifest | ✅ Working |
| zstd / zlib / raw codec selection | ✅ Working |
| SHA-256 per-file and per-block hashing | ✅ Working |
| 256 KiB micro-blocks (configurable) | ✅ Working |
| AI format detection (safetensors, GGUF, LoRA, shards, etc.) | ✅ Working |
| safetensors real tensor metadata (names, shapes, dtypes, offsets) | ✅ Working |
| safetensors shard detection | ✅ Working |
| LoRA/PEFT adapter detection with rank and target modules | ✅ Working |
| GGUF header parsing (version, endianness, tensor count) | ✅ Working |
| Path traversal protection in unpack | ✅ Working |
| Manifest validation (duplicates, codecs, sizes) | ✅ Working |
| Full verification report with block/file hash checks | ✅ Working |
| Benchmark JSON export with environment metadata | ✅ Working |
| Kimari CLI integration adapters | ✅ Working |
| `safetensors` optional dependency support | ✅ Working |
| `zipnn` optional dependency support | ✅ Working |
| Real model benchmarks | 🔜 Planned |
| GGUF block-level compression | 🔬 Research |
| Block-loading (partial decompression) | 🔬 Research |
| Checkpoint/gradients compression | 🔬 Research |

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

# Verify integrity (full report)
kmc verify ./my-model.kmc

# Inspect archive manifest
kmc inspect ./my-model.kmc

# Inspect AI model directory (detects formats, reads tensor metadata)
kmc inspect ./my-model/ --tensors

# Inspect with JSON output
kmc inspect ./my-model/ --json

# Unpack to a directory
kmc unpack ./my-model.kmc ./restored-model/

# Run benchmark with JSON output
kmc bench ./my-model ./my-model-bench.kmc --json --output report.json

# Benchmark with ZipNN comparison
kmc bench ./my-model ./my-model-bench.kmc --compare-zipnn
```

## KMC Archive Format

The `.kmc` format is designed for verifiable, block-oriented storage:

```
┌─────────────────────────────────────┐
│ Magic: "KMC\x00\x01\x00\x00\x00"   │  8 bytes
├─────────────────────────────────────┤
│ Manifest length: uint64 BE          │  8 bytes
├─────────────────────────────────────┤
│ Manifest: JSON (UTF-8 encoded)     │  Variable
│  - version, tool info              │
│  - file entries with paths & hashes│
│  - block entries with codecs       │
│  - tensor entries (v0.3+, optional)│
│  - total sizes and ratios          │
├─────────────────────────────────────┤
│ Block data: concatenated           │  Variable
│  - Each block independently         │
│    compressed (zstd/zlib/raw)      │
│  - SHA-256 verified per block      │
└─────────────────────────────────────┘
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

See [SECURITY_MODEL.md](docs/SECURITY_MODEL.md) for the complete security model.

## Architecture

```
src/kmc/
├── archive.py          # Core pack/unpack/verify with security checks
├── benchmark.py        # Performance benchmarking with codec + ZipNN comparison
├── cli.py              # Command-line interface
├── codecs.py           # Compression codecs (zstd, zlib, raw)
├── formats/
│   ├── __init__.py     # Format module registry
│   ├── safetensors.py  # Safetensors metadata, shards, LoRA detection
│   └── gguf.py         # GGUF header parsing
├── hashing.py          # SHA-256 integrity hashing
├── inspector.py        # AI model format detection with metadata
├── manifest.py         # KMC manifest (JSON metadata with tensor entries)
├── tensor_inspector.py # Legacy safetensors metadata (see formats/safetensors.py)
├── gguf.py             # Legacy GGUF module (see formats/gguf.py)
└── integrations/
    └── kimari.py       # Kimari CLI integration adapters
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Technical Foundation

KMC's approach is informed by research and industry practice:

- **ZipNN** (IBM Research) demonstrates that lossless compression specific to AI models can save ~1/3 of size on popular models, and >50% in some cases, without changing weights.
- **safetensors** (Hugging Face) is treated as the priority format because it's secure, fast, and avoids `pickle` vulnerabilities.
- **GGUF** (llama.cpp) is the standard binary format for quantized models and is planned for future block-level integration.
- **NetZIP** (IBM Research) explores lossless compression for gradients and activations in distributed training — a research direction documented for KMC's roadmap.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Design decisions and module structure
- [Format Specification](docs/FORMAT_SPEC.md) — Complete `.kmc` format spec
- [Security Model](docs/SECURITY_MODEL.md) — Threat model and mitigations
- [Roadmap](docs/ROADMAP.md) — Development priorities
- [Benchmark Plan](docs/BENCHMARK_PLAN.md) — Performance testing strategy
- [Research Notes](docs/RESEARCH_NOTES.md) — Technical references
- [Kimari Integration](docs/KIMARI_INTEGRATION.md) — Integration with Kimari CLI
- [Hugging Face Workflow](docs/HUGGINGFACE_WORKFLOW.md) — Working with Hugging Face models

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

MIT License — see [LICENSE](LICENSE) for details.
