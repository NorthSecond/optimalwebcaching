#!/usr/bin/env python
"""Generate pairwise data with footprint-based cache sizes (0.1% and 10%)."""

import time
import numpy as np

TRACE_PATH = "/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.zst"

# Cache size ratios relative to footprint (unique bytes)
CACHE_RATIOS = [0.001, 0.1]  # 0.1% and 10%


def calculate_footprint(trace) -> int:
    """Calculate footprint (total unique bytes) for the trace."""
    unique_bytes = 0
    for i in range(trace.n_requests):
        if trace.prev_access_idx[i] == -1:  # First access to this object
            unique_bytes += int(trace.obj_sizes[i])
    return unique_bytes


def run_pipeline(max_requests: int, cache_ratio: float, output_path: str):
    """Run FOO pipeline for given trace size and cache ratio."""
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"Testing {max_requests:,} requests, cache_ratio={cache_ratio*100:.1f}%")
    print(f"{'='*60}")

    # 1. Load trace
    print("\n[Step 1] Loading trace...")
    start = time.time()
    from foo_jax.trace_parser import parse_trace
    trace = parse_trace(TRACE_PATH, max_requests=max_requests, use_rust=False)
    print(f"  Requests: {trace.n_requests:,}, Unique: {trace.n_unique_objects:,}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 2. Calculate footprint and cache size
    print("\n[Step 2] Calculating footprint...")
    footprint = calculate_footprint(trace)
    cache_size = int(footprint * cache_ratio)
    print(f"  Footprint: {footprint:,} bytes ({footprint/1024/1024:.2f} MB)")
    print(f"  Cache size ({cache_ratio*100:.1f}%): {cache_size:,} bytes ({cache_size/1024/1024:.2f} MB)")

    # 3. Build topology
    print("\n[Step 3] Building topology...")
    start = time.time()
    from foo_jax.topology import build_topology
    topo = build_topology(trace, cache_size)
    print(f"  Nodes: {topo.n_nodes:,}, Arcs: {topo.n_inner + topo.n_outer:,}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 4. Run solver
    max_iters = min(50000, max(1000, max_requests // 100))
    print(f"\n[Step 4] Running solver ({max_iters} iters)...")
    start = time.time()
    from foo_jax.solver import solve_jit
    solver_result = solve_jit(topo, max_iters=max_iters, tol=1e-4)
    print(f"  Converged: {solver_result.converged}, Iters: {solver_result.iterations}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 5. Extract result
    print("\n[Step 5] Extracting FOO result...")
    start = time.time()
    from foo_jax.output import process_result
    foo_result = process_result(trace, topo, solver_result)
    print(f"  OHR: {foo_result.ohr*100:.2f}%")
    print(f"  Time: {time.time() - start:.1f}s")

    # 6. Analyze dvar distribution
    raw_dvars = foo_result.dvars
    dvars = np.round(raw_dvars).astype(np.int32)
    n_dvar_0 = np.sum(dvars == 0)
    n_dvar_1 = np.sum(dvars == 1)
    print(f"\n[Step 6] Dvar analysis:")
    print(f"  dvar=0 (evict): {n_dvar_0:,} ({100*n_dvar_0/len(dvars):.1f}%)")
    print(f"  dvar=1 (cache): {n_dvar_1:,} ({100*n_dvar_1/len(dvars):.1f}%)")

    # 7. Generate pairwise data
    print("\n[Step 7] Generating pairwise data...")
    start = time.time()
    from foo_jax.pairwise_libcachesim import export_pairwise_libcachesim

    # Adjust sampling for scale
    max_pairs_per_point = 20 if max_requests <= 100000 else 5

    n_pairs = export_pairwise_libcachesim(
        trace, foo_result, output_path,
        cache_size=cache_size,
        max_pairs_per_point=max_pairs_per_point,
        min_history_len=2,
        seed=42,
        sampling_strategy="stratified",
    )
    print(f"  Time: {time.time() - start:.1f}s")

    # Summary
    print(f"\n[Summary]")
    print(f"  Pairs: {n_pairs:,}")
    print(f"  Total time: {time.time() - total_start:.1f}s")

    return n_pairs


def main():
    results = {}

    # Test different scales with different cache ratios
    scales = [100, 10_000, 100_000, 1_000_000]

    for n in scales:
        for ratio in CACHE_RATIOS:
            ratio_str = f"{ratio*100:.1f}".replace(".", "p")
            n_str = f"{n//1000}k" if n >= 1000 else str(n)
            output = f"pairwise_{n_str}_cache{ratio_str}pct.csv"
            key = (n, ratio)

            try:
                pairs = run_pipeline(n, ratio, output)
                results[key] = pairs
            except Exception as e:
                import traceback
                print(f"ERROR at {n} requests, {ratio*100:.1f}% cache: {e}")
                traceback.print_exc()
                results[key] = -1

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Requests':>12} | {'Cache %':>10} | {'Pairs':>12}")
    print("-"*70)
    for (n, ratio), pairs in sorted(results.items()):
        print(f"{n:>12,} | {ratio*100:>9.1f}% | {pairs:>12,}")


if __name__ == "__main__":
    main()
