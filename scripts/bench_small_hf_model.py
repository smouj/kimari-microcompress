#!/usr/bin/env python3
"""Benchmark KMC compression on a small HuggingFace model.

This script compresses a small model from HuggingFace using different
KMC codecs and generates a comparison report. It does NOT download
models automatically — the user must provide the model path.

Suggested small models:
    - sshleifer/tiny-gpt2
    - hf-internal-testing/tiny-random-gpt2

Usage:
    # After downloading a model with:
    #   huggingface-cli download sshleifer/tiny-gpt2 --local-dir ./tiny-gpt2
    python scripts/bench_small_hf_model.py ./tiny-gpt2

Output:
    - Console table with codec comparison
    - JSON file with detailed results
    - Markdown table for documentation

DISCLAIMER: Results are specific to the model and hardware used.
They should not be used as general claims of compression performance.
KMC does NOT reduce inference VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kmc.archive import pack, read_manifest_from_archive, verify
from kmc.codecs.zstd_codec import is_zstd_available


def _format_size(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} bytes"


def _ratio_str(ratio: float) -> str:
    return f"{ratio:.2%}"


def bench_model(model_dir: str, output_dir: str | None = None) -> dict:
    """Benchmark KMC codecs on a model directory."""
    model_path = Path(model_dir).resolve()
    if not model_path.exists():
        print(f"Error: model directory not found: {model_path}")
        sys.exit(1)

    # Calculate original size
    files = [f for f in model_path.rglob("*") if f.is_file()]
    original_size = sum(f.stat().st_size for f in files)
    print(f"Model: {model_path}")
    print(f"Original size: {_format_size(original_size)}")
    print(f"Files: {len(files)}")
    print()

    # Create output directory
    if output_dir is None:
        output_dir = str(model_path.parent / "kmc_bench_output")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Codecs to test
    codecs_to_test = ["auto", "zstd", "zlib", "byteplane", "floatplane"]
    if not is_zstd_available():
        codecs_to_test.remove("zstd")

    results = []
    for codec_name in codecs_to_test:
        archive_path = out_path / f"model_{codec_name}.kmc"
        print(f"Testing codec: {codec_name}...")

        try:
            t0 = time.perf_counter()
            pack(model_path, archive_path, codec=codec_name)
            pack_time = time.perf_counter() - t0

            # Verify
            errors = verify(archive_path)
            if errors:
                print(f"  WARNING: Verify failed: {errors[:2]}")

            comp_size = archive_path.stat().st_size
            ratio = comp_size / original_size if original_size > 0 else 0
            throughput = original_size / pack_time if pack_time > 0 else 0

            # Read manifest for block count
            manifest, _ = read_manifest_from_archive(archive_path)
            n_blocks = sum(len(f.blocks) for f in manifest.files)
            codecs_used = set(b.codec for f in manifest.files for b in f.blocks)

            result = {
                "codec": codec_name,
                "original_size": original_size,
                "compressed_size": comp_size,
                "ratio": ratio,
                "pack_time_s": round(pack_time, 3),
                "throughput_mb_s": round(throughput / 1024 / 1024, 2),
                "blocks": n_blocks,
                "actual_codecs_used": sorted(codecs_used),
                "verify_ok": len(errors) == 0,
            }
            results.append(result)
            print(
                f"  {_format_size(original_size)} -> {_format_size(comp_size)} "
                f"({ratio:.2%}) in {pack_time:.2f}s"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "codec": codec_name,
                    "error": str(e),
                    "verify_ok": False,
                }
            )

    # Summary
    summary = {
        "model_path": str(model_path),
        "original_size": original_size,
        "original_size_human": _format_size(original_size),
        "num_files": len(files),
        "zstd_available": is_zstd_available(),
        "disclaimer": (
            "Results are specific to this model and hardware. "
            "KMC does NOT reduce inference VRAM. "
            "This is a measurement, not a claim of superiority."
        ),
        "codec_results": results,
    }

    # Save JSON
    json_path = out_path / "benchmark_results.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {json_path}")

    # Print markdown table
    print("\n## Codec Comparison\n")
    print("| Codec | Compressed | Ratio | Time (s) | Throughput (MB/s) | Actual Codecs |")
    print("|-------|-----------|-------|----------|-------------------|---------------|")
    for r in results:
        if "error" in r:
            print(f"| {r['codec']} | ERROR | - | - | - | {r['error']} |")
        else:
            print(
                f"| {r['codec']} | {_format_size(r['compressed_size'])} | "
                f"{_ratio_str(r['ratio'])} | {r['pack_time_s']:.3f} | "
                f"{r['throughput_mb_s']:.2f} | {', '.join(r['actual_codecs_used'])} |"
            )

    print(f"\n> **Disclaimer**: {summary['disclaimer']}")
    if original_size < 1_000_000:
        print(">\n> **Synthetic benchmark only** — not representative of real model compression.")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KMC compression on a small HuggingFace model"
    )
    parser.add_argument(
        "model_dir",
        help="Path to the model directory (must be pre-downloaded)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for benchmark artifacts",
    )
    args = parser.parse_args()

    bench_model(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
