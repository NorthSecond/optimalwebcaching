#!/usr/bin/env python
"""
End-to-end test: export-pairs pipeline with real MetaKV trace.
Mimics CLI export-pairs --dvar-file flow without JAX.
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
    n_unique_objects: int
    timestamps: np.ndarray
    obj_ids: np.ndarray
    obj_sizes: np.ndarray

@dataclass
class MockFOOResult:
    dvars: np.ndarray

import types
mock_trace_parser = types.ModuleType('foo_jax.trace_parser')
mock_trace_parser.TraceData = MockTraceData
mock_output = types.ModuleType('foo_jax.output')
mock_output.FOOResult = MockFOOResult
mock_foo_jax = types.ModuleType('foo_jax')
mock_foo_jax.__path__ = ['/home/ubuntu/ssd/optimalwebcaching/foo-jax/foo_jax']
sys.modules['foo_jax'] = mock_foo_jax
sys.modules['foo_jax.trace_parser'] = mock_trace_parser
sys.modules['foo_jax.output'] = mock_output
sys.path.insert(0, '/home/ubuntu/ssd/optimalwebcaching/foo-jax')

from foo_jax.pairwise_libcachesim import (
    export_pairwise_libcachesim, load_dvars_from_cpp_foo
)

TRACE_PATH = '/tmp/test_trace.oracleGeneral.bin'
DVAR_PATH = '/tmp/foo_9947_dvars.txt'
OUTPUT_PATH = '/tmp/e2e_pairs.csv'
CACHE_SIZE = 1007


def parse_binary_trace(path):
    """Parse OracleGeneral binary trace (same as trace_parser.parse_trace_fast)."""
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 24
    ts = np.zeros(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t, oid, sz, _ = struct.unpack_from('<IQIq', data, i * 24)
        ts[i], ids[i], sizes[i] = t, oid, sz
    unique = set()
    for i in range(n):
        unique.add((int(ids[i]), int(sizes[i])))
    return MockTraceData(
        n_requests=n,
        n_unique_objects=len(unique),
        timestamps=ts,
        obj_ids=ids,
        obj_sizes=sizes,
    )


def main():
    print("=" * 70)
    print("End-to-End Test: export-pairs pipeline")
    print("=" * 70)

    # Step 1: Parse trace
    t0 = time.time()
    trace = parse_binary_trace(TRACE_PATH)
    print(f"\n1. Parsed trace: {trace.n_requests:,} requests, "
          f"{trace.n_unique_objects:,} unique objects ({time.time()-t0:.2f}s)")

    # Step 2: Load dvars from C++ FOO
    t0 = time.time()
    dvars = load_dvars_from_cpp_foo(DVAR_PATH, trace)
    dvars_int = np.round(dvars).astype(int)
    n_hi = np.sum(dvars_int == 1)
    n_lo = np.sum(dvars_int == 0)
    print(f"2. Loaded dvars: hi={n_hi:,} ({100*n_hi/len(dvars):.1f}%), "
          f"lo={n_lo:,} ({100*n_lo/len(dvars):.1f}%) ({time.time()-t0:.2f}s)")

    # Step 3: Create FOOResult (same as CLI)
    foo_result = MockFOOResult(dvars=dvars)

    # Step 4: Export pairwise data
    print(f"3. Generating pairwise data (cache={CACHE_SIZE}, stratified)...")
    t0 = time.time()
    n_pairs = export_pairwise_libcachesim(
        trace=trace,
        foo_result=foo_result,
        output_path=OUTPUT_PATH,
        cache_size=CACHE_SIZE,
        max_pairs_per_point=100,
        min_history_len=2,
        seed=42,
        sampling_strategy="stratified",
    )
    gen_time = time.time() - t0

    # Step 5: Validate output
    print(f"\n4. Validation:")
    if n_pairs > 0 and os.path.exists(OUTPUT_PATH):
        file_size = os.path.getsize(OUTPUT_PATH)
        with open(OUTPUT_PATH) as f:
            header = f.readline().strip().split(',')
            first_row = f.readline().strip().split(',')
            # Count total lines
            n_lines = 2
            for _ in f:
                n_lines += 1

        print(f"   File: {OUTPUT_PATH} ({file_size/1024:.1f} KB)")
        print(f"   Columns: {len(header)} (expected 19)")
        print(f"   Data rows: {n_lines - 1} (expected {n_pairs})")
        print(f"   First row cols: {len(first_row)}")

        assert len(header) == 19, f"Column count mismatch: {len(header)}"
        assert n_lines - 1 == n_pairs, f"Row count mismatch: {n_lines-1} vs {n_pairs}"
        assert len(first_row) == 19, f"First row col count: {len(first_row)}"

        print(f"\n   ✓ All validations passed")
        print(f"   ✓ {n_pairs:,} pairs generated in {gen_time:.1f}s")
        print(f"   ✓ Throughput: {trace.n_requests/gen_time:,.0f} req/s")
    else:
        print(f"   ✗ No pairs generated!")
        return False

    print(f"\n{'='*70}")
    print("END-TO-END TEST PASSED ✓")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
