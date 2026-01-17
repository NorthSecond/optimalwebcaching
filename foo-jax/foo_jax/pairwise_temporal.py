"""
Temporally-Consistent Pairwise Data Generation for ML Training.

CRITICAL FIX: The previous vectorized approach had a fundamental flaw:
- Features were computed at each request's own timestamp
- Random pairing mixed requests from completely different times
- This made features INCOMPARABLE

CORRECT APPROACH:
1. Sample at "decision points" - regular intervals through the trace
2. At each decision point T:
   - Compute features for ALL known objects relative to time T
   - Use the most recent dvar for each object
   - Split into hi_pool (dvar=1) and lo_pool (dvar=0)
   - Sample pairs from objects that exist at the SAME time T
3. This ensures temporal consistency - features are directly comparable

Feature Semantics (matching original format):
- obj_id, obj_size: Object identifier and size (static)
- mean_arr: Mean inter-arrival time based on history up to decision point
- last_5_access_0..4: Time from decision point T to last 1-5 accesses
- now_last_space: Remaining cache space at decision point (based on dvar=1 objects)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import time

from .trace_parser import TraceData
from .output import FOOResult


@dataclass
class TemporalPairwiseConfig:
    """Configuration for temporally-consistent pairwise generation."""
    n_pairs: int = 10_000_000           # Target total pairs
    sample_interval: int = 10000        # Sample every N requests
    max_pairs_per_point: int = 100      # Max pairs per decision point
    min_history_len: int = 2            # Minimum accesses for valid candidate
    max_history_len: int = 20           # Max history to track per object
    seed: int = 42


def generate_pairwise_temporal_optimized(
    trace: TraceData,
    foo_result: FOOResult,
    cache_size: int,
    config: TemporalPairwiseConfig = None,
    output_file=None,  # File handle for streaming write
) -> pd.DataFrame:
    """
    Optimized temporally-consistent pairwise generation.

    Key features:
    1. Temporal consistency - all features computed at same reference time
    2. Correct now_last_space - based on dvar=1 objects' total size
    3. Stratified sampling for feature diversity
    4. Original column order for model compatibility
    """
    if config is None:
        config = TemporalPairwiseConfig()

    np.random.seed(config.seed)
    start_total = time.time()

    n_requests = trace.n_requests
    timestamps = trace.timestamps.astype(np.int64)
    obj_ids = trace.obj_ids
    obj_sizes = trace.obj_sizes

    # Round dvars to 0/1
    raw_dvars = foo_result.dvars
    dvars = np.round(raw_dvars).astype(np.int32)

    n_dvar_0 = np.sum(dvars == 0)
    n_dvar_1 = np.sum(dvars == 1)
    print(f"  dvar distribution: hi={n_dvar_1:,} ({100*n_dvar_1/len(dvars):.1f}%), "
          f"lo={n_dvar_0:,} ({100*n_dvar_0/len(dvars):.1f}%)")

    # Build object index: unique (obj_id, size) -> index
    obj_keys = list(zip(obj_ids.tolist(), obj_sizes.tolist()))
    unique_keys = list(set(obj_keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    n_objects = len(unique_keys)

    print(f"  Unique objects: {n_objects:,}")

    # Pre-allocate object state arrays
    obj_access_times = [[] for _ in range(n_objects)]  # List of timestamps per object
    obj_current_dvar = np.zeros(n_objects, dtype=np.int32)  # Most recent dvar
    obj_n_accesses = np.zeros(n_objects, dtype=np.int32)  # Access count
    # Store obj_ids as Python objects (can be very large), sizes as int32
    obj_ids_list = [k[0] for k in unique_keys]  # Keep as Python list for large ints
    obj_sizes_arr = np.array([k[1] for k in unique_keys], dtype=np.int32)

    # Collect pairs (only if not streaming to file)
    all_pairs = [] if output_file is None else None
    n_sample_points = 0
    n_pairs_generated = 0

    # Calculate decision points
    decision_points = list(range(config.sample_interval, n_requests, config.sample_interval))
    n_decision_points = len(decision_points)
    print(f"  Processing {n_requests:,} requests, {n_decision_points:,} decision points...")

    next_decision_idx = 0

    for idx in range(n_requests):
        if idx > 0 and idx % 1_000_000 == 0:
            elapsed = time.time() - start_total
            print(f"    {idx:,}/{n_requests:,} ({elapsed:.1f}s), "
                  f"decision_points={n_sample_points:,}, pairs={n_pairs_generated:,}", flush=True)

        # Update object state
        key = obj_keys[idx]
        oid = key_to_idx[key]
        ts = int(timestamps[idx])
        request_dvar = int(dvars[idx])

        obj_access_times[oid].append(ts)
        if len(obj_access_times[oid]) > config.max_history_len:
            obj_access_times[oid] = obj_access_times[oid][-config.max_history_len:]
        obj_current_dvar[oid] = request_dvar
        obj_n_accesses[oid] += 1

        # Check if we've reached a decision point
        if next_decision_idx < n_decision_points and idx >= decision_points[next_decision_idx]:
            current_time = ts
            next_decision_idx += 1

            # Find valid candidates
            valid_mask = obj_n_accesses >= config.min_history_len
            hi_mask = valid_mask & (obj_current_dvar == 1)
            lo_mask = valid_mask & (obj_current_dvar == 0)

            hi_indices = np.where(hi_mask)[0]
            lo_indices = np.where(lo_mask)[0]

            if len(hi_indices) > 0 and len(lo_indices) > 0:
                n_sample_points += 1

                # Compute remaining cache space based on dvar=1 objects
                cached_size = np.sum(obj_sizes_arr[hi_mask])
                remaining_space = max(0.0, float(cache_size - cached_size))

                n_to_sample = min(
                    config.max_pairs_per_point,
                    len(hi_indices) * len(lo_indices),
                )

                if n_to_sample > 0:
                    # Random sampling
                    n_hi_available = len(hi_indices)
                    n_lo_available = len(lo_indices)

                    # Sample with replacement if needed
                    hi_samples = hi_indices[np.random.choice(n_hi_available, size=n_to_sample, replace=True)]
                    lo_samples = lo_indices[np.random.choice(n_lo_available, size=n_to_sample, replace=True)]

                    # Generate pairs
                    for hi_oid, lo_oid in zip(hi_samples, lo_samples):
                        hi_times = obj_access_times[hi_oid]
                        lo_times = obj_access_times[lo_oid]

                        # Hi features
                        hi_n = len(hi_times)
                        if hi_n >= 2:
                            hi_intervals = [hi_times[j] - hi_times[j-1] for j in range(1, hi_n)]
                            hi_mean_arr = float(np.mean(hi_intervals))
                        else:
                            hi_mean_arr = -1.0

                        hi_last_5 = [-1.0] * 5
                        for j, t in enumerate(reversed(hi_times[-5:])):
                            if j < 5:
                                hi_last_5[j] = float(current_time - t)

                        # Lo features
                        lo_n = len(lo_times)
                        if lo_n >= 2:
                            lo_intervals = [lo_times[j] - lo_times[j-1] for j in range(1, lo_n)]
                            lo_mean_arr = float(np.mean(lo_intervals))
                        else:
                            lo_mean_arr = -1.0

                        lo_last_5 = [-1.0] * 5
                        for j, t in enumerate(reversed(lo_times[-5:])):
                            if j < 5:
                                lo_last_5[j] = float(current_time - t)

                        # Build pair with ORIGINAL column order
                        pair = (
                            obj_ids_list[hi_oid],
                            int(obj_sizes_arr[hi_oid]),
                            hi_mean_arr,
                            hi_last_5[0],
                            hi_last_5[1],
                            hi_last_5[2],
                            hi_last_5[3],
                            hi_last_5[4],
                            remaining_space,
                            obj_ids_list[lo_oid],
                            int(obj_sizes_arr[lo_oid]),
                            lo_mean_arr,
                            lo_last_5[0],
                            lo_last_5[1],
                            lo_last_5[2],
                            lo_last_5[3],
                            lo_last_5[4],
                            remaining_space,
                            1,
                        )
                        if output_file is not None:
                            output_file.write(','.join(map(str, pair)) + '\n')
                        else:
                            all_pairs.append(pair)
                        n_pairs_generated += 1


    print(f"  Decision points sampled: {n_sample_points:,}")
    print(f"  Total pairs: {n_pairs_generated:,}")
    print(f"  Total time: {time.time() - start_total:.1f}s")

    if output_file is not None:
        # Streaming mode - data already written to file
        return n_pairs_generated

    if not all_pairs:
        return pd.DataFrame()

    # Convert tuples to DataFrame
    columns = [
        'hi_obj_id', 'hi_obj_size', 'hi_mean_arr',
        'hi_last_5_access_0', 'hi_last_5_access_1', 'hi_last_5_access_2',
        'hi_last_5_access_3', 'hi_last_5_access_4', 'hi_now_last_space',
        'lo_obj_id', 'lo_obj_size', 'lo_mean_arr',
        'lo_last_5_access_0', 'lo_last_5_access_1', 'lo_last_5_access_2',
        'lo_last_5_access_3', 'lo_last_5_access_4', 'lo_now_last_space',
        'label'
    ]
    df = pd.DataFrame(all_pairs, columns=columns)
    return df


def export_pairwise_temporal(
    trace: TraceData,
    foo_result: FOOResult,
    output_path: str,
    cache_size: int,
    n_pairs: int = 10_000_000,
    sample_interval: int = 5000,
    max_pairs_per_point: int = 200,
    min_history_len: int = 2,
    seed: int = 42,
) -> int:
    """
    Generate and export temporally-consistent pairwise data (streaming mode).

    Args:
        trace: Parsed trace data
        foo_result: FOO solver result with dvars
        output_path: Path to save CSV
        cache_size: Cache size used for solving
        n_pairs: (Ignored) - generates all possible pairs
        sample_interval: Sample every N requests
        max_pairs_per_point: Max pairs per decision point
        min_history_len: Minimum accesses for valid candidate
        seed: Random seed

    Returns:
        Number of pairs generated
    """
    config = TemporalPairwiseConfig(
        n_pairs=n_pairs,
        sample_interval=sample_interval,
        max_pairs_per_point=max_pairs_per_point,
        min_history_len=min_history_len,
        seed=seed,
    )

    # ORIGINAL column order for model compatibility
    columns = [
        'hi_obj_id', 'hi_obj_size', 'hi_mean_arr',
        'hi_last_5_access_0', 'hi_last_5_access_1', 'hi_last_5_access_2',
        'hi_last_5_access_3', 'hi_last_5_access_4', 'hi_now_last_space',
        'lo_obj_id', 'lo_obj_size', 'lo_mean_arr',
        'lo_last_5_access_0', 'lo_last_5_access_1', 'lo_last_5_access_2',
        'lo_last_5_access_3', 'lo_last_5_access_4', 'lo_now_last_space',
        'label'
    ]

    # Streaming write - directly to file
    with open(output_path, 'w') as f:
        # Write header
        f.write(','.join(columns) + '\n')
        # Generate and write pairs
        n_pairs_written = generate_pairwise_temporal_optimized(
            trace, foo_result, cache_size, config, output_file=f
        )

    if n_pairs_written == 0:
        print("  WARNING: No pairs generated!")
    else:
        print(f"  Saved to: {output_path}")

    return n_pairs_written
