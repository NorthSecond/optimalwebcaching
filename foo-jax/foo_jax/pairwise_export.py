"""
Generate pairwise preference data for learning FOO's optimal eviction policy.

Key insight: Each REQUEST has its own dvar representing whether to cache
from this access to the next. An object's priority changes over time:
- Hot object (high dvar) can become cold (low dvar) as access pattern changes
- We track the CURRENT dvar (from most recent access) for each cached object

Important: FOO's dvars should be 0 or 1 (integer solution):
- dvar = 1: cache this object (will be accessed again soon)
- dvar = 0: evict this object (last access or long gap)

Objects with dvar=0 at their LAST access have rich history and are perfect
lo candidates for pairwise learning!

Features per object:
- obj_id: object identifier
- obj_size: size in bytes
- mean_arr: mean inter-arrival time (-1 if < 2 accesses)
- last_5_access_0-4: time intervals to last 5 accesses
- now_last_space: remaining cache space at decision time
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict

from .trace_parser import TraceData
from .output import FOOResult


@dataclass
class ObjectState:
    """State of an object in cache."""
    obj_id: int
    obj_size: int
    access_times: List[int] = field(default_factory=list)
    current_dvar: int = 0  # dvar from MOST RECENT access: 0 or 1


def compute_features(state: ObjectState, current_time: int, remaining_space: int) -> dict:
    """Compute features for an object."""
    history = state.access_times

    # Mean inter-arrival time (requires >= 2 accesses)
    if len(history) >= 2:
        intervals = [history[i] - history[i-1] for i in range(1, len(history))]
        mean_arr = float(np.mean(intervals))
    else:
        mean_arr = -1.0

    # Last 5 access intervals (time from current to past accesses)
    last_5 = [-1.0] * 5
    if history:
        for i, h in enumerate(reversed(history[-5:])):
            if i < 5:
                last_5[i] = float(current_time - h)

    return {
        'obj_id': state.obj_id,
        'obj_size': state.obj_size,
        'mean_arr': mean_arr,
        'last_5_access_0': last_5[0],
        'last_5_access_1': last_5[1],
        'last_5_access_2': last_5[2],
        'last_5_access_3': last_5[3],
        'last_5_access_4': last_5[4],
        'now_last_space': float(remaining_space)
    }


def generate_pairwise_data(
    trace: TraceData,
    foo_result: FOOResult,
    cache_size: int,
    sample_ratio: float = 0.3,
    max_pairs_per_point: int = 20,
    min_history_len: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    基于FOO决策变量直接采样pairwise数据（不依赖缓存模拟）。

    核心思路：
    - 遍历trace，维护每个对象的访问历史
    - 在每个时间点，找到有历史的对象中：
      - hi候选：当前dvar=1（FOO认为应该缓存）
      - lo候选：当前dvar=0（FOO认为应该驱逐）
    - 随机采样hi-lo对

    这样不依赖缓存大小产生冲突，直接学习FOO的决策模式。
    """
    np.random.seed(seed)
    n_requests = trace.n_requests

    # 四舍五入dvars到0/1
    raw_dvars = foo_result.dvars
    dvars = np.round(raw_dvars).astype(np.int32)

    n_dvar_0 = np.sum(dvars == 0)
    n_dvar_1 = np.sum(dvars == 1)
    n_fractional = np.sum((raw_dvars > 0.01) & (raw_dvars < 0.99))
    print(f"  Dvar distribution: 0={n_dvar_0:,}, 1={n_dvar_1:,}, fractional={n_fractional:,}")

    # 对象状态：obj_key -> ObjectState
    objects: Dict[Tuple[int, int], ObjectState] = {}
    pairs = []
    n_sample_points = 0
    n_sampled = 0
    n_skipped_no_hi = 0
    n_skipped_no_lo = 0

    for idx in range(n_requests):
        if idx > 0 and idx % 100000 == 0:
            print(f"    Progress: {idx:,}/{n_requests:,}, sample_points: {n_sample_points:,}, "
                  f"pairs: {len(pairs):,}", flush=True)

        obj_id = int(trace.obj_ids[idx])
        obj_size = int(trace.obj_sizes[idx])
        timestamp = int(trace.timestamps[idx])
        key = (obj_id, obj_size)
        request_dvar = int(dvars[idx])

        # 更新或创建对象状态
        if key not in objects:
            objects[key] = ObjectState(obj_id=obj_id, obj_size=obj_size,
                                       access_times=[timestamp], current_dvar=request_dvar)
        else:
            objects[key].access_times.append(timestamp)
            if len(objects[key].access_times) > 20:
                objects[key].access_times = objects[key].access_times[-20:]
            objects[key].current_dvar = request_dvar

        # 采样：在有足够对象时随机采样
        if len(objects) >= 10 and np.random.random() < sample_ratio:
            # 找有历史的hi和lo候选
            hi_cands = [v for v in objects.values()
                       if v.current_dvar == 1 and len(v.access_times) >= min_history_len]
            lo_cands = [v for v in objects.values()
                       if v.current_dvar == 0 and len(v.access_times) >= min_history_len]

            if hi_cands and lo_cands:
                n_sample_points += 1
                n_sampled += 1
                n_sample = min(max_pairs_per_point, len(hi_cands) * len(lo_cands))

                # 估算剩余空间（用于特征）
                remaining_space = max(0, cache_size - sum(v.obj_size for v in objects.values()))

                for _ in range(n_sample):
                    hi_state = hi_cands[np.random.randint(len(hi_cands))]
                    lo_state = lo_cands[np.random.randint(len(lo_cands))]

                    hi_f = compute_features(hi_state, timestamp, remaining_space)
                    lo_f = compute_features(lo_state, timestamp, remaining_space)

                    pairs.append({
                        'hi_obj_id': hi_f['obj_id'], 'hi_obj_size': hi_f['obj_size'],
                        'hi_mean_arr': hi_f['mean_arr'],
                        'hi_last_5_access_0': hi_f['last_5_access_0'],
                        'hi_last_5_access_1': hi_f['last_5_access_1'],
                        'hi_last_5_access_2': hi_f['last_5_access_2'],
                        'hi_last_5_access_3': hi_f['last_5_access_3'],
                        'hi_last_5_access_4': hi_f['last_5_access_4'],
                        'hi_now_last_space': hi_f['now_last_space'],
                        'lo_obj_id': lo_f['obj_id'], 'lo_obj_size': lo_f['obj_size'],
                        'lo_mean_arr': lo_f['mean_arr'],
                        'lo_last_5_access_0': lo_f['last_5_access_0'],
                        'lo_last_5_access_1': lo_f['last_5_access_1'],
                        'lo_last_5_access_2': lo_f['last_5_access_2'],
                        'lo_last_5_access_3': lo_f['last_5_access_3'],
                        'lo_last_5_access_4': lo_f['last_5_access_4'],
                        'lo_now_last_space': lo_f['now_last_space'],
                        'label': 1
                    })
            else:
                if not hi_cands:
                    n_skipped_no_hi += 1
                if not lo_cands:
                    n_skipped_no_lo += 1

    print(f"  Sample points: {n_sample_points:,}, Sampled: {n_sampled:,}, "
          f"Skipped (no_hi={n_skipped_no_hi:,}, no_lo={n_skipped_no_lo:,}), Pairs: {len(pairs):,}")
    return pd.DataFrame(pairs)


def export_pairwise_csv(
    trace: TraceData,
    foo_result: FOOResult,
    output_path: str,
    cache_size: int = None,
    sample_ratio: float = 0.5,
    max_pairs_per_point: int = 30,
    min_history_len: int = 2,
    seed: int = 42,
    max_file_size_gb: float = 2.0
) -> pd.DataFrame:
    """生成并导出pairwise数据，支持自动分片。"""
    if cache_size is None:
        unique_bytes = sum(int(trace.obj_sizes[i]) for i in range(trace.n_requests)
                          if trace.prev_access_idx[i] == -1)
        cache_size = unique_bytes // 10
        print(f"  Cache size: {cache_size:,} bytes")

    df = generate_pairwise_data(trace, foo_result, cache_size, sample_ratio,
                                 max_pairs_per_point, min_history_len, seed)

    if len(df) == 0:
        print("  WARNING: No pairs! Try: smaller cache_size or check dvar distribution")
        return df

    columns = ['hi_obj_id', 'hi_obj_size', 'hi_mean_arr',
               'hi_last_5_access_0', 'hi_last_5_access_1', 'hi_last_5_access_2',
               'hi_last_5_access_3', 'hi_last_5_access_4', 'hi_now_last_space',
               'lo_obj_id', 'lo_obj_size', 'lo_mean_arr',
               'lo_last_5_access_0', 'lo_last_5_access_1', 'lo_last_5_access_2',
               'lo_last_5_access_3', 'lo_last_5_access_4', 'lo_now_last_space', 'label']
    df = df[columns]

    # 估算CSV大小：每行约200字节
    estimated_size_gb = len(df) * 200 / (1024**3)
    print(f"  Estimated CSV size: {estimated_size_gb:.2f} GB")

    if estimated_size_gb <= max_file_size_gb:
        # 单文件输出
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    else:
        # 分片输出
        rows_per_shard = int(len(df) * max_file_size_gb / estimated_size_gb)
        n_shards = (len(df) + rows_per_shard - 1) // rows_per_shard
        print(f"  Splitting into {n_shards} shards (~{rows_per_shard:,} rows each)")

        base_path = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'csv'

        for i in range(n_shards):
            start_idx = i * rows_per_shard
            end_idx = min((i + 1) * rows_per_shard, len(df))
            shard_path = f"{base_path}_shard{i:03d}.{ext}"
            df.iloc[start_idx:end_idx].to_csv(shard_path, index=False)
            print(f"    Shard {i}: {shard_path} ({end_idx - start_idx:,} rows)")

    return df
