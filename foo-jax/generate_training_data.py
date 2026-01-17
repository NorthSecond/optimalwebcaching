#!/usr/bin/env python
"""
Generate GB-scale temporally-consistent pairwise training data.

Key improvements:
1. Temporal consistency - features computed at same reference time
2. Correct now_last_space - based on dvar=1 objects' cache occupancy
3. Stratified sampling - for feature diversity
4. Original column format - compatible with existing model
"""

import time
import os
import sys
import numpy as np
import argparse

TRACE_PATH = "/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.zst"


def calculate_footprint(trace) -> int:
    """Calculate footprint (total unique bytes) for the trace."""
    unique_bytes = 0
    for i in range(trace.n_requests):
        if trace.prev_access_idx[i] == -1:
            unique_bytes += int(trace.obj_sizes[i])
    return unique_bytes


def validate_data_quality(df) -> dict:
    """Validate the quality of generated pairwise data."""
    print("\n[Data Quality Validation]")

    stats = {}
    stats['n_pairs'] = len(df)

    # Feature columns to check
    feature_cols = ['obj_size', 'mean_arr', 'last_5_access_0', 'now_last_space']

    print(f"  Pairs: {stats['n_pairs']:,}")
    print()
    print(f"  {'Feature':<20} {'Hi Mean':>12} {'Hi Std':>12} {'Lo Mean':>12} {'Lo Std':>12}")
    print("  " + "-" * 70)

    for feat in feature_cols:
        hi_col = f'hi_{feat}'
        lo_col = f'lo_{feat}'

        if hi_col in df.columns and lo_col in df.columns:
            hi_vals = df[hi_col][df[hi_col] >= 0]
            lo_vals = df[lo_col][df[lo_col] >= 0]

            hi_mean = hi_vals.mean() if len(hi_vals) > 0 else 0
            hi_std = hi_vals.std() if len(hi_vals) > 0 else 0
            lo_mean = lo_vals.mean() if len(lo_vals) > 0 else 0
            lo_std = lo_vals.std() if len(lo_vals) > 0 else 0

            print(f"  {feat:<20} {hi_mean:>12.2f} {hi_std:>12.2f} {lo_mean:>12.2f} {lo_std:>12.2f}")

    # Check now_last_space specifically
    if 'hi_now_last_space' in df.columns:
        space_vals = df['hi_now_last_space']
        print(f"\n  now_last_space: min={space_vals.min():.0f}, max={space_vals.max():.0f}, "
              f"mean={space_vals.mean():.0f}, std={space_vals.std():.0f}")

    # Size distribution analysis
    if 'hi_obj_size' in df.columns and 'lo_obj_size' in df.columns:
        hi_sizes = df['hi_obj_size']
        lo_sizes = df['lo_obj_size']
        print(f"\n  Size percentiles (hi): p10={hi_sizes.quantile(0.1):.0f}, "
              f"p50={hi_sizes.quantile(0.5):.0f}, p90={hi_sizes.quantile(0.9):.0f}")
        print(f"  Size percentiles (lo): p10={lo_sizes.quantile(0.1):.0f}, "
              f"p50={lo_sizes.quantile(0.5):.0f}, p90={lo_sizes.quantile(0.9):.0f}")

    return stats


def run_single_config(
    max_requests: int,
    cache_ratio: float,
    n_pairs: int,
    output_dir: str,
    trace_cache: dict = None,
    sample_interval: int = 5000,
    max_pairs_per_point: int = 200,
) -> int:
    """Run FOO pipeline for a single configuration."""
    total_start = time.time()

    ratio_str = f"{cache_ratio*100:.1f}".replace(".", "p")
    n_str = f"{max_requests//1_000_000}M" if max_requests >= 1_000_000 else f"{max_requests//1000}k"
    output_path = os.path.join(output_dir, f"pairwise_{n_str}_cache{ratio_str}pct.csv")

    print(f"\n{'='*70}")
    print(f"Config: {n_str} requests, {cache_ratio*100:.1f}% cache")
    print(f"Target: {n_pairs:,} pairs")
    print(f"Sample interval: {sample_interval:,}, pairs/point: {max_pairs_per_point}")
    print(f"Output: {output_path}")
    print(f"{'='*70}")

    # Check if already exists
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path)
        if existing_size > 1024 * 1024:  # > 1MB
            print(f"  SKIP: Output exists ({existing_size/1024/1024:.1f} MB)")
            with open(output_path) as f:
                lines = sum(1 for _ in f) - 1
            return lines
        else:
            print(f"  Removing incomplete file...")
            os.remove(output_path)

    # 1. Load trace
    print("\n[1/5] Loading trace...")
    start = time.time()

    if trace_cache and max_requests in trace_cache:
        trace = trace_cache[max_requests]
        print(f"  Using cached trace")
    else:
        from foo_jax.trace_parser import parse_trace
        trace = parse_trace(TRACE_PATH, max_requests=max_requests, use_rust=False)
        if trace_cache is not None:
            trace_cache[max_requests] = trace

    print(f"  Requests: {trace.n_requests:,}, Unique: {trace.n_unique_objects:,}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 2. Calculate footprint and cache size
    print("\n[2/5] Calculating footprint...")
    footprint = calculate_footprint(trace)
    cache_size = int(footprint * cache_ratio)
    print(f"  Footprint: {footprint:,} bytes ({footprint/1024/1024:.2f} MB)")
    print(f"  Cache ({cache_ratio*100:.1f}%): {cache_size:,} bytes ({cache_size/1024/1024:.2f} MB)")

    # 3. Build topology
    print("\n[3/5] Building topology...")
    start = time.time()
    from foo_jax.topology import build_topology
    topo = build_topology(trace, cache_size)
    print(f"  Nodes: {topo.n_nodes:,}, Arcs: {topo.n_inner + topo.n_outer:,}")
    print(f"  Time: {time.time() - start:.1f}s")

    # 4. Run solver
    max_iters = min(100000, max(10000, max_requests // 50))
    print(f"\n[4/5] Running solver (max {max_iters:,} iters)...")
    start = time.time()
    from foo_jax.solver import solve_jit
    solver_result = solve_jit(topo, max_iters=max_iters, tol=1e-4)
    print(f"  Converged: {solver_result.converged}, Iters: {solver_result.iterations}")
    print(f"  Time: {time.time() - start:.1f}s")

    # Extract result
    from foo_jax.output import process_result
    foo_result = process_result(trace, topo, solver_result)
    print(f"  OHR: {foo_result.ohr*100:.2f}%")

    # Dvar analysis
    dvars = np.round(foo_result.dvars).astype(np.int32)
    n_hi = np.sum(dvars == 1)
    n_lo = np.sum(dvars == 0)
    print(f"  dvar=1 (cache): {n_hi:,} ({100*n_hi/len(dvars):.1f}%)")
    print(f"  dvar=0 (evict): {n_lo:,} ({100*n_lo/len(dvars):.1f}%)")

    # 5. Generate pairwise data
    print(f"\n[5/5] Generating pairwise data...")
    start = time.time()
    from foo_jax.pairwise_temporal import export_pairwise_temporal

    n_pairs_generated = export_pairwise_temporal(
        trace, foo_result, output_path,
        cache_size=cache_size,
        n_pairs=n_pairs,
        sample_interval=sample_interval,
        max_pairs_per_point=max_pairs_per_point,
        min_history_len=2,
        seed=42 + int(cache_ratio * 1000),
    )
    gen_time = time.time() - start
    print(f"  Generation time: {gen_time:.1f}s")

    # Summary
    total_time = time.time() - total_start
    print(f"\n[Summary]")
    print(f"  Pairs: {n_pairs_generated:,}")
    print(f"  Total time: {total_time/60:.1f} minutes")

    if n_pairs_generated > 0 and os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"  File size: {file_size/1024/1024:.1f} MB")

    return n_pairs_generated


def main():
    parser = argparse.ArgumentParser(description="Generate GB-scale pairwise data")
    parser.add_argument("--max-requests", type=int, default=10_000_000,
                       help="Max requests per config (default: 10M)")
    parser.add_argument("--pairs-per-config", type=int, default=10_000_000,
                       help="Target pairs per config (default: 10M)")
    parser.add_argument("--output-dir", type=str, default="data_gb",
                       help="Output directory (default: data_gb)")
    parser.add_argument("--cache-ratios", type=str, default="0.001,0.01,0.05,0.10",
                       help="Comma-separated cache ratios")
    parser.add_argument("--sample-interval", type=int, default=5000,
                       help="Sample every N requests (default: 5000)")
    parser.add_argument("--max-pairs-per-point", type=int, default=200,
                       help="Max pairs per decision point (default: 200)")
    args = parser.parse_args()

    cache_ratios = [float(r) for r in args.cache_ratios.split(",")]

    # Calculate expected data
    n_decision_points = args.max_requests // args.sample_interval
    max_pairs_possible = n_decision_points * args.max_pairs_per_point
    actual_pairs_per_config = min(args.pairs_per_config, max_pairs_possible)

    print("=" * 70)
    print("Pairwise Data Generation (Temporal Consistency)")
    print("=" * 70)
    print(f"Trace: {TRACE_PATH}")
    print(f"Max requests: {args.max_requests:,}")
    print(f"Pairs per config: {args.pairs_per_config:,}")
    print(f"Sample interval: {args.sample_interval:,}")
    print(f"Max pairs/point: {args.max_pairs_per_point}")
    print(f"Decision points: ~{n_decision_points:,}")
    print(f"Max achievable: ~{max_pairs_possible:,}")
    print(f"Cache ratios: {[f'{r*100:.1f}%' for r in cache_ratios]}")
    print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    trace_cache = {}
    results = {}
    total_start = time.time()

    for i, ratio in enumerate(cache_ratios):
        print(f"\n\n{'#'*70}")
        print(f"# Configuration {i+1}/{len(cache_ratios)}: {ratio*100:.1f}% cache")
        print(f"{'#'*70}")

        try:
            n_pairs = run_single_config(
                args.max_requests, ratio, args.pairs_per_config,
                args.output_dir, trace_cache,
                sample_interval=args.sample_interval,
                max_pairs_per_point=args.max_pairs_per_point,
            )
            results[ratio] = n_pairs
        except Exception as e:
            import traceback
            print(f"\nERROR at {ratio*100:.1f}% cache: {e}")
            traceback.print_exc()
            results[ratio] = -1

    # Final summary
    total_time = time.time() - total_start
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print()
    print(f"{'Cache %':>10} | {'Pairs':>15} | {'Est. Size':>12} | {'Status'}")
    print("-" * 55)

    total_pairs = 0
    for ratio in sorted(results.keys()):
        pairs = results[ratio]
        if pairs > 0:
            total_pairs += pairs
            est_mb = pairs * 150 / 1e6
            print(f"{ratio*100:>9.1f}% | {pairs:>15,} | {est_mb:>10.1f} MB | ✓")
        elif pairs == 0:
            print(f"{ratio*100:>9.1f}% | {'SKIPPED':>15} | {'-':>12} | ⊘")
        else:
            print(f"{ratio*100:>9.1f}% | {'FAILED':>15} | {'-':>12} | ✗")

    print("-" * 55)
    total_gb = total_pairs * 150 / 1e9
    print(f"{'TOTAL':>10} | {total_pairs:>15,} | {total_gb:>10.2f} GB |")

    # List output files
    print("\nOutput files:")
    output_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.csv')])
    total_size = 0
    for f in output_files:
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {f}: {size/1024/1024:.1f} MB")
    print(f"  Total: {total_size/1024/1024/1024:.2f} GB")


if __name__ == "__main__":
    main()
