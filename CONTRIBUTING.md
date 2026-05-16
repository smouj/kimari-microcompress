# Contributing to Kimari MicroCompress

Thank you for your interest in contributing to KMC! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional. We are all working toward the same goal: making AI model storage and transfer more efficient.

## How to Contribute

### Reporting Issues

1. Search existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Include:
   - KMC version (`kmc --help` shows the version)
   - Python version (`python --version`)
   - OS and architecture
   - Minimal reproduction steps
   - Expected vs actual behavior

### Submitting Changes

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/my-feature`
3. **Make changes** following the code style below
4. **Add tests** for any new functionality
5. **Run the full validation suite** (see below)
6. **Commit** with a descriptive message
7. **Push** to your fork
8. **Open a Pull Request** against `main`

### Validation Checklist

Before submitting a PR, ensure all of the following pass:

```bash
# Tests
pytest -q

# Linting
ruff check .

# Formatting
ruff format --check .

# CLI sanity check
python -m kmc --help
python -m kmc pack --help
python -m kmc inspect --help
```

All 228+ tests must pass with no linting or formatting errors.

## Code Style

### Python

- **Python 3.10+** with type hints on all public functions
- **PEP 8** enforced by ruff
- **Line length**: 100 characters maximum
- **Docstrings** for all public functions, classes, and modules (Google style)
- **Dataclasses** for structured data
- **`pathlib.Path`** over `os.path`
- **`from __future__ import annotations`** in all modules

### Import Order

Enforced by ruff (isort-compatible):

1. Standard library
2. Third-party
3. Local/application

### Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

## Architecture Guidelines

### Module Organization

- `src/kmc/` — Core library
  - `archive.py` — Pack/unpack/verify operations
  - `manifest.py` — KMC manifest data structures
  - `codecs/` — Compression codec subpackage
  - `formats/` — Format-specific parsers (safetensors, GGUF)
  - `workflows/` — Artifact-specific workflows (LoRA, checkpoint)
  - `integrations/` — External tool integrations (Kimari)
- `tests/` — Test suite
- `docs/` — Documentation
- `scripts/` — Utility scripts
- `examples/` — Usage examples

### Adding a New Codec

1. Create `src/kmc/codecs/my_codec.py`
2. Implement the `Codec` protocol from `codecs/base.py`
3. Register in `codecs/registry.py`
4. Add selector logic in `codecs/selector.py` (if dtype-aware)
5. Add tests in `tests/test_v05_codecs.py` (or new test file)
6. Update `FORMAT_SPEC.md` with codec_metadata schema

### Adding a New Format Parser

1. Create `src/kmc/formats/my_format.py`
2. Implement detection and metadata extraction
3. Register in `formats/__init__.py`
4. Add workflow in `workflows/` if needed
5. Add CLI flag in `cli.py` and `inspector.py`
6. Add manifest metadata schema in `FORMAT_SPEC.md`
7. Add comprehensive tests

### Adding a New Workflow

1. Create `src/kmc/workflows/my_workflow.py`
2. Implement detection and manifest metadata building
3. Add CLI subcommand in `cli.py`
4. Add inspector flag in `inspector.py`
5. Add Kimari integration adapter in `integrations/kimari.py`
6. Add documentation in `docs/`
7. Add comprehensive tests

## Testing Guidelines

### Test Structure

- `test_roundtrip.py` — Core pack/unpack roundtrip tests
- `test_manifest.py` — Manifest serialization/deserialization
- `test_security.py` — Security-related tests (path traversal, etc.)
- `test_safetensors.py` — Safetensors format tests
- `test_gguf.py` — GGUF format tests
- `test_lora_workflow.py` — LoRA workflow tests
- `test_checkpoint_workflow.py` — Checkpoint workflow tests
- `test_v03_features.py` — v0.3 feature regression tests
- `test_v04_codecs.py` — v0.4 codec tests
- `test_v05_manifest.py` — v0.5 manifest tests
- `test_v05_cli.py` — v0.5 CLI tests
- `test_benchmark_inspector.py` — Benchmark and inspector tests

### Writing Tests

- Use `pytest` with `tmp_path` fixture for temporary files
- Test both success and error cases
- Test edge cases (empty files, large files, malformed input)
- Include backward compatibility tests for older manifest versions
- Use the `tests/fixtures/tensors.py` helpers for synthetic tensor data
- Each new feature must have its own test module or extend the appropriate existing module

### Test Naming

```python
def test_pack_lora_with_valid_adapter(tmp_path):
    ...

def test_pack_lora_without_adapter_model(tmp_path):
    ...

def test_pack_lora_incomplete_config(tmp_path):
    ...
```

## Documentation Guidelines

- All features must be documented in the appropriate `docs/` file
- Update `README.md` when adding user-facing features
- Update `FORMAT_SPEC.md` when changing the archive format
- Update `ROADMAP.md` when completing roadmap items
- Use clear, honest language — no hype or unverified claims
- Include code examples for all new APIs
- Document limitations explicitly

## Strict Rules

These rules are non-negotiable:

1. **No pickle deserialization** — Never use `pickle.load()` or `torch.load()` on any data. Pickle-based files must only be detected by name and compressed as raw bytes.
2. **Lossless only** — Every byte must be perfectly preserved. No exceptions.
3. **No weight modification** — Never alter, quantize, prune, or modify model data.
4. **No invented benchmarks** — All benchmark claims must be from real models with reproducible methodology. Synthetic benchmarks must be clearly labeled.
5. **No VRAM claims** — KMC does not reduce inference VRAM. Never imply otherwise.
6. **Optional heavy dependencies** — Features requiring external packages (safetensors, zipnn) must degrade gracefully with clear warnings when those packages are not installed.
7. **Backward compatibility** — New versions must be able to read archives from older versions.
8. **Security first** — Path traversal, zip bombs, and manifest bombs must be mitigated.

## Release Process

1. Update version in `pyproject.toml` and `src/kmc/__init__.py`
2. Update `CHANGELOG.md`
3. Update `docs/ROADMAP.md` — mark completed items
4. Run full validation suite
5. Commit with message: `feat: KMC v0.X.Y — description`
6. Tag: `git tag v0.X.Y`
7. Push: `git push origin main --tags`
8. Create GitHub Release with changelog notes

## Getting Help

- Open a GitHub issue for bugs or feature requests
- Check existing documentation in `docs/`
- Review the architecture document for design context
