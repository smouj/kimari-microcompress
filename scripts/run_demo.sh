#!/bin/bash
# Demo script for Kimari MicroCompress
# Creates a synthetic model, packs it, verifies, unpacks, and benchmarks.

set -e

echo "=== KMC Demo ==="
echo

# Step 1: Create demo model
echo "--- Creating demo model ---"
python scripts/create_demo_model.py
echo

# Step 2: Pack
echo "--- Packing ---"
kmc pack demo-model demo-model.kmc
echo

# Step 3: Verify
echo "--- Verifying ---"
kmc verify demo-model.kmc
echo

# Step 4: Inspect
echo "--- Inspecting ---"
kmc inspect demo-model.kmc
echo

# Step 5: Unpack
echo "--- Unpacking ---"
kmc unpack demo-model.kmc restored/
echo

# Step 6: Verify roundtrip
echo "--- Verifying roundtrip ---"
if diff -r demo-model restored/ > /dev/null 2>&1; then
    echo "Roundtrip OK: original and restored files are identical."
else
    echo "Roundtrip FAILED: files differ!"
    exit 1
fi
echo

# Step 7: Benchmark
echo "--- Benchmark ---"
kmc bench demo-model demo-model-bench.kmc
echo

echo "=== Demo complete ==="
