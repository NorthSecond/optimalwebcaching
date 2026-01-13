"""
Pairwise dataset export for learning-to-cache.

Extracts (hi, lo) pairs from FOO solution where:
- hi: objects that FOO decided to KEEP (high priority)
- lo: objects that FOO decided to EVICT (low priority)

Features include:
- obj_id, size
- mean_arr (mean arrival interval)
- now_last_space (time since last access)
- last_5_access_0..4 (last 5 access time deltas)

Output format: CSV with columns:
    hi_obj_id, hi_size, hi_mean_arr, hi_now_last_space,
    hi_last_5_access_0, ..., hi_last_5_access_4,
    lo_obj_id, lo_size, lo_mean_arr, lo_now_last_space,
    lo_last_5_access_0, ..., lo_last_5_access_4,
    label (always 1, indicating hi > lo)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .trace_parser import TraceData
from .topology import Topology
from .output import FOOResult


@dataclass
class ObjectFeatures:
    """Features for a single object at a given time."""
    obj_id: int
    size: int
    mean_arr: float  # Mean inter-arrival time
    now_last_space: float  # Time since last access
    last_5_access: List[float]  # Last 5 access time deltas


@dataclass
class PairwiseRecord:
    """A (hi, lo) pair for learning."""
    hi: ObjectFeatures
    lo: ObjectFeatures
    time_idx: int  # When this decision was made


def compute_features(
    trace: TraceData,
    precompute_last_5: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute features for all requests in the trace.

    Returns:
        mean_arr: Mean inter-arrival time per request, shape (n_requests,)
        now_last_space: Time since last access per request, shape (n_requests,)
        last_5_access: Last 5 access deltas per request, shape (n_requests, 5)
    """
    n = trace.n_requests
    timestamps = trace.timestamps.astype(np.float64)

    # Compute now_last_space (time since previous access of same object)
    now_last_space = np.zeros(n, dtype=np.float32)

    # Track per-object access history
    # Using dict for simplicity; could be optimized with numpy for large traces
    obj_history = {}  # (obj_id, size) -> list of (index, timestamp)

    for i in range(n):
        key = (int(trace.obj_ids[i]), int(trace.obj_sizes[i]))

        if key not in obj_history:
            obj_history[key] = []

        history = obj_history[key]

        if len(history) > 0:
            # Time since last access
            last_ts = history[-1][1]
            now_last_space[i] = timestamps[i] - last_ts
        else:
            now_last_space[i] = 0.0  # First access

        history.append((i, timestamps[i]))

    # Compute mean_arr (mean inter-arrival time up to this point)
    mean_arr = np.zeros(n, dtype=np.float32)
    obj_history_reset = {}

    for i in range(n):
        key = (int(trace.obj_ids[i]), int(trace.obj_sizes[i]))

        if key not in obj_history_reset:
            obj_history_reset[key] = []

        history = obj_history_reset[key]
        history.append(timestamps[i])

        if len(history) > 1:
            # Mean of inter-arrival times
            deltas = np.diff(history)
            mean_arr[i] = float(np.mean(deltas))
        else:
            mean_arr[i] = 0.0

    # Compute last_5_access (last 5 access time deltas)
    last_5_access = np.zeros((n, 5), dtype=np.float32)

    if precompute_last_5:
        obj_history_5 = {}

        for i in range(n):
            key = (int(trace.obj_ids[i]), int(trace.obj_sizes[i]))

            if key not in obj_history_5:
                obj_history_5[key] = []

            history = obj_history_5[key]

            # Compute deltas from previous accesses
            if len(history) > 0:
                deltas = []
                for j in range(min(5, len(history))):
                    delta = timestamps[i] - history[-(j+1)]
                    deltas.append(delta)

                # Pad with zeros if less than 5 previous accesses
                while len(deltas) < 5:
                    deltas.append(0.0)

                last_5_access[i] = deltas[:5]

            history.append(timestamps[i])

    return mean_arr, now_last_space, last_5_access


def identify_conflict_points(
    result: FOOResult,
    topo: Topology,
    threshold_hi: float = 0.7,
    threshold_lo: float = 0.3
) -> List[Tuple[int, List[int], List[int]]]:
    """
    Identify time points where caching decisions conflict.

    A conflict point is where some objects are kept (dvar > threshold_hi)
    while others are evicted (dvar < threshold_lo).

    Args:
        result: FOO result with dvars
        topo: Network topology
        threshold_hi: dvar threshold for "keep" decision
        threshold_lo: dvar threshold for "evict" decision

    Returns:
        List of (time_idx, hi_indices, lo_indices) tuples
    """
    conflicts = []

    # Get outer arc info
    outer_trace_idx = np.asarray(topo.outer_trace_idx)
    dvars = result.dvars

    # Find indices with high and low dvars (among outer arcs)
    hi_mask = dvars[outer_trace_idx] > threshold_hi
    lo_mask = dvars[outer_trace_idx] < threshold_lo

    hi_indices = np.where(hi_mask)[0]
    lo_indices = np.where(lo_mask)[0]

    if len(hi_indices) > 0 and len(lo_indices) > 0:
        # Return as single conflict point (simplified)
        # In practice, we'd identify specific time windows
        conflicts.append((0, list(outer_trace_idx[hi_indices]), list(outer_trace_idx[lo_indices])))

    return conflicts


def extract_pairs(
    trace: TraceData,
    result: FOOResult,
    topo: Topology,
    max_pairs: int = 10000,
    threshold_hi: float = 0.7,
    threshold_lo: float = 0.3,
    seed: int = 42
) -> List[PairwiseRecord]:
    """
    Extract (hi, lo) pairs from FOO solution.

    Args:
        trace: Original trace data
        result: FOO result
        topo: Network topology
        max_pairs: Maximum number of pairs to extract
        threshold_hi: dvar threshold for "keep" decision
        threshold_lo: dvar threshold for "evict" decision
        seed: Random seed for sampling

    Returns:
        List of PairwiseRecord
    """
    np.random.seed(seed)

    # Precompute features
    mean_arr, now_last_space, last_5_access = compute_features(trace)

    # Get indices of hi and lo decisions
    outer_trace_idx = np.asarray(topo.outer_trace_idx)
    dvars = result.dvars

    hi_indices = outer_trace_idx[dvars[outer_trace_idx] > threshold_hi]
    lo_indices = outer_trace_idx[dvars[outer_trace_idx] < threshold_lo]

    if len(hi_indices) == 0 or len(lo_indices) == 0:
        return []

    # Sample pairs
    n_pairs = min(max_pairs, len(hi_indices) * len(lo_indices))
    pairs = []

    for _ in range(n_pairs):
        hi_idx = int(np.random.choice(hi_indices))
        lo_idx = int(np.random.choice(lo_indices))

        # Create features
        hi_feat = ObjectFeatures(
            obj_id=int(trace.obj_ids[hi_idx]),
            size=int(trace.obj_sizes[hi_idx]),
            mean_arr=float(mean_arr[hi_idx]),
            now_last_space=float(now_last_space[hi_idx]),
            last_5_access=list(last_5_access[hi_idx])
        )

        lo_feat = ObjectFeatures(
            obj_id=int(trace.obj_ids[lo_idx]),
            size=int(trace.obj_sizes[lo_idx]),
            mean_arr=float(mean_arr[lo_idx]),
            now_last_space=float(now_last_space[lo_idx]),
            last_5_access=list(last_5_access[lo_idx])
        )

        pairs.append(PairwiseRecord(hi=hi_feat, lo=lo_feat, time_idx=0))

    return pairs


def write_pairs_csv(
    path: str,
    pairs: List[PairwiseRecord]
) -> None:
    """
    Write pairs to CSV file.

    Format: hi_obj_id, hi_size, hi_mean_arr, hi_now_last_space,
            hi_last_5_access_0, ..., hi_last_5_access_4,
            lo_obj_id, lo_size, lo_mean_arr, lo_now_last_space,
            lo_last_5_access_0, ..., lo_last_5_access_4,
            label
    """
    with open(path, 'w') as f:
        # Header
        cols = []
        for prefix in ['hi', 'lo']:
            cols.extend([
                f'{prefix}_obj_id',
                f'{prefix}_size',
                f'{prefix}_mean_arr',
                f'{prefix}_now_last_space',
            ])
            for i in range(5):
                cols.append(f'{prefix}_last_5_access_{i}')
        cols.append('label')

        f.write(','.join(cols) + '\n')

        # Data rows
        for pair in pairs:
            row = []

            for feat in [pair.hi, pair.lo]:
                row.extend([
                    str(feat.obj_id),
                    str(feat.size),
                    f'{feat.mean_arr:.2f}',
                    f'{feat.now_last_space:.2f}',
                ])
                for delta in feat.last_5_access:
                    row.append(f'{delta:.2f}')

            row.append('1')  # Label: hi > lo

            f.write(','.join(row) + '\n')
