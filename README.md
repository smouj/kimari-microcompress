# Kimari MicroCompress (KMC)

**Reversible lossless compression for AI models: safetensors, GGUF, checkpoints, LoRA/QLoRA and future block-loading.**

[![CI](https://github.com/kimaridb/kimari-microcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/kimaridb/kimari-microcompress/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Kimari MicroCompress (KMC) is an experimental tool for **lossless, reversible compression** of AI model files. It focuses on **storage, transfer, verification, and packaging** without modifying the original weights. The approach is grounded in the observation that AI model files — particularly `safetensors` and quantized formats — contain significant redundancy that general-purpose compression tools don't exploit optimally.

**Key principle:** Every byte that goes in must come out identically. KMC provides byte-exact roundtrip integrity verified via SHA-256 hashes at both the file and block level.

## Features

| Feature | Status |
|---------|--------|
| `kmc pack` — Compress files/directories | ✅ Working |
| `kmc unpack` — Decompress archives | ✅ Working |
| `kmc verify` — Verify archive integrity | ✅ Working |
| `kmc inspect` — View archive manifest | ✅ Working |
| `kmc bench` — Benchmark performance | ✅ Working |
| `.kmc` archive format with JSON manifest | ✅ Working |
| zstd / zlib / raw codec selection | ✅ Working |
| SHA-256 per-file and per-block hashing | ✅ Working |
| 256 KiB micro-blocks (configurable) | ✅ Working |
| AI format detection (safetensors, GGUF, .bin, .pt, .ckpt) | ✅ Working |
| safetensors tensor metadata parsing | ✅ Working |
| GGUF header parsing | ✅ Working |
| Path traversal protection in unpack | ✅ Working |
| Real model benchmarks | 🔜 Planned |
| `kimari compress` integration | 🔜 Planned |
| GGUF block-level compression | 🔜 Planned |
| Block-loading (partial decompression) | 🔬 Research |
| Checkpoint/gradients compression | 🔬 Research |

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/kimaridb/kimari-microcompress.git
cd kimari-microcompress
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- `zstandard` (recommended, for best compression)
- `zlib` (built-in, used as fallback)

## Quick Start

```bash
# Pack a model directory
kmc pack ./my-model ./my-model.kmc

# Verify integrity
kmc verify ./my-model.kmc

# Inspect the archive
kmc inspect ./my-model.kmc

# Unpack to a directory
kmc unpack ./my-model.kmc ./restored-model/

# Run benchmark
kmc bench ./my-model ./my-model-bench.kmc
```

## KMC Archive Format

The `.kmc` format is designed for verifiable, block-oriented storage:

```
┌─────────────────────────────────────┐
│ Magic: "KMC\x00\x01\x00\x00\x00"   │  8 bytes
├─────────────────────────────────────┤
│ Manifest length: uint64 BE          │  8 bytes
├─────────────────────────────────────┤
│ Manifest: JSON (UTF-8)             │  Variable
│  - version, tool info              │
│  - file entries with paths & hashes│
│  - block entries with codecs       │
│  - total sizes and ratios          │
├─────────────────────────────────────┤
│ Block data: concatenated           │  Variable
│  - Each block independently         │
│    compressed (zstd/zlib/raw)      │
│  - SHA-256 verified per block      │
└─────────────────────────────────────┘
```

See [FORMAT_SPEC.md](docs/FORMAT_SPEC.md) for the complete specification.

## Architecture

```
src/kmc/
├── archive.py          # Core pack/unpack/verify operations
├── benchmark.py        # Performance benchmarking
├── cli.py              # Command-line interface
├── codecs.py           # Compression codecs (zstd, zlib, raw)
├── gguf.py             # GGUF format support (future)
├── hashing.py          # SHA-256 integrity hashing
├── inspector.py        # AI model format detection
├── manifest.py         # KMC manifest (JSON metadata)
└── tensor_inspector.py # safetensors metadata parsing
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
- [Kimari Integration](docs/KIMARI_INTEGRATION.md) — Future `kimari compress` integration

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -q

# Lint
ruff check .

# Create demo model and test
python scripts/create_demo_model.py
bash scripts/run_demo.sh
```

## License

MIT License — see [LICENSE](LICENSE) for details.
