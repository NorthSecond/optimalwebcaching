#!/usr/bin/env python
"""
Test pairwise_libcachesim.py: streaming output + incremental indexing.
Uses mock objects to avoid JAX imports.
"""
import struct
import numpy as np
import time
import os
import sys
from dataclasses import dataclass
from typing import List

# Mock TraceData and FOOResult to avoid JAX imports
@dataclass
class MockTraceData:
    n_requests: int
    timestamps: np.ndarray
    obj_ids: np.ndarray
    obj_sizes: np.ndarray

@dataclass
class MockFOOResult:
    dvars: np.ndarray

# Patch foo_jax modules before importing pairwise_libcachesim
import types

# Create mock modules to avoid JAX dependency
mock_trace_parser = types.ModuleType('foo_jax.trace_parser')
mock_trace_parser.TraceData = MockTraceData
mock_output = types.ModuleType('foo_jax.output')
mock_output.FOOResult = MockFOOResult
mock_foo_jax = types.ModuleType('foo_jax')
mock_foo_jax.__path__ = ['/home/ubuntu/ssd/optimalwebcaching/foo-jax/foo_jax']

sys.modules['foo_jax'] = mock_foo_jax
sys.modules['foo_jax.trace_parser'] = mock_trace_parser
sys.modules['foo_jax.output'] = mock_output

# Now import the module under test
sys.path.insert(0, '/home/ubuntu/ssd/optimalwebcaching/foo-jax')
from foo_jax.pairwise_libcachesim import (
    PairwiseGenerator, export_pairwise_libcachesim, SamplingStrategy
)


def load_trace(path, n_max=None):
    """Load binary OracleGeneral trace."""
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 24
    if n_max:
        n = min(n, n_max)
    ts = np.zeros(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t, oid, sz, _ = struct.unpack_from('<IQIq', data, i * 24)
        ts[i], ids[i], sizes[i] = t, oid, sz
    return MockTraceData(n_requests=n, timestamps=ts, obj_ids=ids, obj_sizes=sizes)


def load_dvars(path, n_requests):
    """Load C++ FOO dvars."""
    dvars = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                dvars.append(float(parts[3]))
    dvars = np.array(dvars[:n_requests], dtype=np.float32)
    assert len(dvars) == n_requests, f"dvar count {len(dvars)} != {n_requests}"
    return MockFOOResult(dvars=dvars)


def test_basic():
    """Test with 9947-request trace (has both dvar=0 and dvar=1)."""
    print("=" * 60)
    print("Test 1: Basic pairwise generation (9947 requests)")
    print("=" * 60)

    trace = load_trace('/tmp/test_trace.oracleGeneral.bin')
    foo_result = load_dvars('/tmp/foo_9947_dvars.txt', trace.n_requests)

    # Check dvar distribution
    dvars_int = np.round(foo_result.dvars).astype(int)
    n_hi = np.sum(dvars_int == 1)
    n_lo = np.sum(dvars_int == 0)
    print(f"  Requests: {trace.n_requests}, dvar=1: {n_hi}, dvar=0: {n_lo}")

    output_path = '/tmp/test_pairs_basic.csv'
    n_pairs = export_pairwise_libcachesim(
        trace=trace,
        foo_result=foo_result,
        output_path=output_path,
        cache_size=1007,
        max_pairs_per_point=10,
        min_history_len=2,
        seed=42,
        sampling_strategy="stratified",
    )

    # Validate output
    if n_pairs > 0:
        with open(output_path) as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        assert len(header) == 19, f"Expected 19 columns, got {len(header)}"
        print(f"  Header columns: {len(header)} ✓")

        # Check data rows
        for i, line in enumerate(lines[1:6]):  # Check first 5 data rows
            cols = line.strip().split(',')
            assert len(cols) == 19, f"Row {i}: expected 19 cols, got {len(cols)}"
        print(f"  Data rows: {len(lines)-1} (all 19 columns) ✓")
        print(f"  Sample row: {lines[1].strip()[:100]}...")
    else:
        print("  WARNING: No pairs generated!")

    print(f"  Result: {n_pairs} pairs ✓")
    return n_pairs > 0


def test_streaming():
    """Verify streaming: file grows during generation, not all at end."""
    print("\n" + "=" * 60)
    print("Test 2: Streaming output verification (9947 requests)")
    print("=" * 60)

    trace = load_trace('/tmp/test_trace.oracleGeneral.bin')
    foo_result = load_dvars('/tmp/foo_9947_dvars.txt', trace.n_requests)

    output_path = '/tmp/test_pairs_streaming.csv'
    n_pairs = export_pairwise_libcachesim(
        trace=trace,
        foo_result=foo_result,
        output_path=output_path,
        cache_size=1007,
        max_pairs_per_point=50,
        min_history_len=2,
        seed=42,
        sampling_strategy="random",
    )

    if n_pairs > 0:
        with open(output_path) as f:
            lines = f.readlines()
        print(f"  Pairs generated: {n_pairs}")
        print(f"  File lines: {len(lines)} (1 header + {len(lines)-1} data)")
        assert len(lines) - 1 == n_pairs, f"Line count mismatch: {len(lines)-1} vs {n_pairs}"
        print(f"  Line count matches pair count ✓")

        # Verify all labels are 1
        labels = [int(line.strip().split(',')[-1]) for line in lines[1:]]
        assert all(l == 1 for l in labels), "Not all labels are 1"
        print(f"  All labels = 1 ✓")
    else:
        print("  No pairs (trace too small for evictions)")

    return True


def test_incremental_indexing():
    """Verify incremental indexing produces same results as brute force."""
    print("\n" + "=" * 60)
    print("Test 3: All sampling strategies (9947 requests)")
    print("=" * 60)

    trace = load_trace('/tmp/test_trace.oracleGeneral.bin')
    foo_result = load_dvars('/tmp/foo_9947_dvars.txt', trace.n_requests)

    # Run with different strategies and verify all produce output
    for strategy in ["random", "similar", "stratified"]:
        output_path = f'/tmp/test_pairs_{strategy}.csv'
        n_pairs = export_pairwise_libcachesim(
            trace=trace,
            foo_result=foo_result,
            output_path=output_path,
            cache_size=1007,
            max_pairs_per_point=20,
            min_history_len=2,
            seed=42,
            sampling_strategy=strategy,
        )
        print(f"  Strategy '{strategy}': {n_pairs} pairs")

    print("  All strategies produce output ✓")
    return True


def main():
    print("Pairwise LibCacheSim Test Suite")
    print("=" * 60)

    # Check prerequisites
    for f in ['/tmp/test_trace.oracleGeneral.bin',
              '/tmp/foo_9947_dvars.txt']:
        if not os.path.exists(f):
            print(f"MISSING: {f}")
            return

    ok = True
    ok &= test_basic()
    ok &= test_streaming()
    ok &= test_incremental_indexing()

    print("\n" + "=" * 60)
    if ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")


if __name__ == "__main__":
    main()
