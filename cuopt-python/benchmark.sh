#!/bin/bash
source .venv/bin/activate

echo "=== FOO cuOpt Performance Benchmark ==="
echo "Testing different problem sizes with Dual Simplex (CPU)"
echo ""

for size in 10000 100000 1000000 10000000; do
    if [ -f "test_cluster54_${size}.dat" ]; then
        echo "--- Size: $size requests ---"
        time python foo_cuopt_datamodel.py "test_cluster54_${size}.dat" 1073741824 /tmp/bench_${size}.txt --solver DualSimplex 2>&1 | grep -E "(Loaded|Graph:|Unique|Status:|Objective:|ExLPcuopt)"
        echo ""
    fi
done
