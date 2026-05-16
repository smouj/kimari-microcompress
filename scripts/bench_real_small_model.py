#!/usr/bin/env python3
"""Benchmark KMC compression with a small real model.

Usage:
    python scripts/bench_real_small_model.py ./models/tiny-gpt2 --output reports/tiny-gpt2

This script requires a locally downloaded model. It does NOT download
models automatically.

If the model path does not exist, it prints an error and exits.

Output:
    - JSON report with all benchmark results
    - Markdown table printed to stdout
    - Environment information for reproducibility

No invented benchmarks. All results come from actual local runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _run_cmd(cmd: list[str], timeout: int = 300) -> tuple[int, str, float]:
    """Run a command and return (returncode, stdout, elapsed_seconds)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - t0
        return result.returncode, result.stdout, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return -1, f"Timeout after {timeout}s", elapsed
    except FileNotFoundError:
        elapsed = time.perf_counter() - t0
        return -2, f"Command not found: {cmd[0]}", elapsed


def _get_environment() -> dict:
    """Collect environment information."""
    env = {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "os_name": platform.system(),
        "os_version": platform.release(),
        "cpu": platform.processor() or "unknown",
        "kmc_version": "0.5.0-alpha",
    }

    # Check for zstd
    try:
        import zstandard  # noqa: F401

        env["zstd_available"] = True
    except ImportError:
        env["zstd_available"] = False

    # Check for safetensors
    try:
        import safetensors  # noqa: F401

        env["safetensors_available"] = True
    except ImportError:
        env["safetensors_available"] = False

    # Check for ZipNN
    try:
        import zipnn  # noqa: F401

        env["zipnn_available"] = True
    except ImportError:
        env["zipnn_available"] = False

    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KMC with a small real model (must be downloaded locally)"
    )
    parser.add_argument("model_path", help="Path to the locally downloaded model")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path prefix for reports (default: ./reports/<model_name>)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        print("Download a tiny model manually first.")
        print("Example: huggingface-cli download gpt2 --local-dir ./models/gpt2")
        sys.exit(1)

    model_name = model_path.name
    output_prefix = args.output or f"reports/{model_name}"
    output_dir = Path(output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _get_environment()

    # Run inspections and benchmarks
    results: dict = {
        "model_path": str(model_path),
        "model_name": model_name,
        "environment": env,
        "benchmarks": {},
        "errors": [],
    }

    # 1. Inspect the model
    print(f"=== KMC Real Model Benchmark: {model_name} ===")
    print()

    print("Step 1: Inspecting model...")
    rc, stdout, elapsed = _run_cmd(
        ["python", "-m", "kmc", "inspect", str(model_path), "--json", "--tensors"]
    )
    if rc == 0:
        try:
            results["inspection"] = json.loads(stdout)
        except json.JSONDecodeError:
            results["inspection"] = {"raw": stdout}
        print("  OK")
    else:
        results["errors"].append(f"Inspect failed: {stdout[:200]}")
        print(f"  FAILED: {stdout[:100]}")

    # 2. Run benchmarks with different codecs
    codec_configs = [
        ("auto", True, "auto-tensor-aware"),
        ("zstd", False, "zstd"),
        ("byteplane", True, "byteplane-tensor-aware"),
        ("floatplane", True, "floatplane-tensor-aware"),
    ]

    for codec, tensor_aware, label in codec_configs:
        output_file = output_dir / f"{model_name}.{label}.kmc"
        print(f"\nStep: Benchmarking with codec={codec}, tensor_aware={tensor_aware}...")

        cmd = [
            "python",
            "-m",
            "kmc",
            "bench",
            str(model_path),
            str(output_file),
            "--codec",
            codec,
            "--json",
        ]
        if tensor_aware:
            cmd.append("--tensor-aware")

        rc, stdout, elapsed = _run_cmd(cmd)
        if rc == 0:
            try:
                bench_data = json.loads(stdout)
                results["benchmarks"][label] = bench_data
                ratio = bench_data.get("kmc_ratio", 0)
                print(f"  OK: ratio={ratio:.2%}, time={elapsed:.1f}s")
            except json.JSONDecodeError:
                results["errors"].append(f"Benchmark {label}: JSON parse error")
                print("  JSON parse error")
        else:
            results["errors"].append(f"Benchmark {label}: {stdout[:200]}")
            print(f"  FAILED: {stdout[:100]}")

    # 3. Optional ZipNN comparison
    if env.get("zipnn_available"):
        print("\nStep: ZipNN comparison...")
        output_file = output_dir / f"{model_name}.zipnn-compare.kmc"
        rc, stdout, elapsed = _run_cmd(
            [
                "python",
                "-m",
                "kmc",
                "bench",
                str(model_path),
                str(output_file),
                "--codec",
                "auto",
                "--tensor-aware",
                "--compare-zipnn",
                "--json",
            ]
        )
        if rc == 0:
            try:
                bench_data = json.loads(stdout)
                results["benchmarks"]["zipnn-compare"] = bench_data
                print("  OK")
            except json.JSONDecodeError:
                results["errors"].append("ZipNN comparison: JSON parse error")

    # Save JSON report
    report_file = output_dir / f"{model_name}.kmc-bench.json"
    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {report_file}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Codec':<25} {'Ratio':>8} {'Pack(s)':>10} {'Compressed':>14}")
    print("-" * 60)
    for label, bench_data in results.get("benchmarks", {}).items():
        ratio = bench_data.get("kmc_ratio", 0)
        pack_time = bench_data.get("kmc_pack_time", 0)
        comp_size = bench_data.get("kmc_compressed_size", 0)
        print(f"{label:<25} {ratio:>7.2%} {pack_time:>10.2f} {comp_size:>14,}")

    if results.get("errors"):
        print("\nErrors encountered:")
        for err in results["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
