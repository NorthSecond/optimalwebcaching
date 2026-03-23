"""Pointwise candidate export using libCacheSim decision points."""

from __future__ import annotations

from typing import Any, IO, Optional
import time

import libcachesim as lcs
import numpy as np

from .output import FOOResult
from .pairwise_libcachesim import ObjectState, load_dvars_from_cpp_foo
from .trace_parser import TraceData


class PointwiseGenerator:
    """Export cached candidates at eviction points with soft FOO labels."""

    def __init__(
        self,
        cache_size: int,
        trace: TraceData,
        foo_result: FOOResult,
        max_samples_per_eviction: Optional[int] = None,
        min_history_len: int = 2,
        seed: int = 42,
    ) -> None:
        self.cache_size = cache_size
        self.trace = trace
        self.soft_dvars = np.asarray(foo_result.dvars, dtype=np.float32)
        self.max_samples_per_eviction = max_samples_per_eviction
        self.min_history_len = min_history_len
        self.seed = seed

        self.object_states: dict[tuple[int, int], ObjectState] = {}
        self._output_file: Optional[IO[str]] = None
        self._rng = np.random.default_rng(seed)

        self.current_req_idx = 0
        self.n_evictions = 0
        self.n_decision_points = 0
        self.n_rows_generated = 0

        self._lru: dict[int, int] = {}
        self._lru_counter = 0
        self._cache_obj_ids: dict[tuple[int, int], int] = {}
        self._cache_obj_id_to_key: dict[int, tuple[int, int]] = {}
        self._cached_keys: set[tuple[int, int]] = set()
        self._eligible_cached_keys: dict[tuple[int, int], None] = {}

    def _get_object_key(self, obj_id: int, obj_size: int) -> tuple[int, int]:
        return (obj_id, obj_size)

    def _get_cache_obj_id(self, key: tuple[int, int]) -> int:
        cache_obj_id = self._cache_obj_ids.get(key)
        if cache_obj_id is not None:
            return cache_obj_id

        cache_obj_id = len(self._cache_obj_ids) + 1
        self._cache_obj_ids[key] = cache_obj_id
        self._cache_obj_id_to_key[cache_obj_id] = key
        return cache_obj_id

    def _next_access_target(self, state: ObjectState) -> float:
        if not state.access_vtimes:
            return 0.0

        last_access_idx = state.access_vtimes[-1]
        next_idx = int(self.trace.next_access_idx[last_access_idx])
        if next_idx < 0:
            return 0.0
        return float(self.soft_dvars[next_idx])

    def _build_row(
        self,
        state: ObjectState,
        *,
        remaining_space: float,
    ) -> tuple[float | int, ...]:
        last_5 = state.compute_last_5_access(self.current_req_idx)
        return (
            state.obj_id,
            state.obj_size,
            state.compute_mean_arr(),
            last_5[0],
            last_5[1],
            last_5[2],
            last_5[3],
            last_5[4],
            remaining_space,
            self._next_access_target(state),
        )

    def _sample_candidate_keys(self, candidate_keys: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if (
            self.max_samples_per_eviction is None
            or len(candidate_keys) <= self.max_samples_per_eviction
        ):
            return candidate_keys
        sample_idx = self._rng.choice(
            len(candidate_keys),
            size=self.max_samples_per_eviction,
            replace=False,
        )
        sample_idx.sort()
        return [candidate_keys[int(idx)] for idx in sample_idx]

    def _init_hook(self, params: Any) -> dict[str, Any]:
        return {}

    def _hit_hook(self, data: dict[str, Any], req: lcs.Request) -> None:
        pass

    def _miss_hook(self, data: dict[str, Any], req: lcs.Request) -> None:
        pass

    def _eviction_hook(self, data: dict[str, Any], req: lcs.Request) -> int:
        self.n_evictions += 1
        remaining_space = self.cache_size - self.cache.get_occupied_byte()

        candidate_keys = list(self._eligible_cached_keys)
        candidate_keys = self._sample_candidate_keys(candidate_keys)

        if candidate_keys and self._output_file is not None:
            self.n_decision_points += 1
            rows = [
                ",".join(
                    map(
                        str,
                        self._build_row(
                            self.object_states[key],
                            remaining_space=remaining_space,
                        ),
                    )
                )
                for key in candidate_keys
            ]
            self._output_file.write("\n".join(rows) + "\n")
            self.n_rows_generated += len(candidate_keys)

        if self._lru:
            lru_obj_id = min(self._lru, key=lambda obj_id: self._lru[obj_id])
            del self._lru[lru_obj_id]
            return lru_obj_id

        return req.obj_id

    def _remove_hook(self, data: dict[str, Any], obj_id: int) -> None:
        self._lru.pop(obj_id, None)
        key = self._cache_obj_id_to_key.get(obj_id)
        if key is not None:
            self._cached_keys.discard(key)
            self._eligible_cached_keys.pop(key, None)

    def _free_hook(self, data: dict[str, Any]) -> None:
        self._lru.clear()
        self._cached_keys.clear()
        self._eligible_cached_keys.clear()

    def generate(self, output_file: IO[str]) -> int:
        self._output_file = output_file
        start_time = time.time()
        n_requests = self.trace.n_requests

        print(f"  Processing {n_requests:,} requests...")

        self.cache = lcs.PluginCache(
            cache_size=self.cache_size,
            cache_init_hook=self._init_hook,
            cache_hit_hook=self._hit_hook,
            cache_miss_hook=self._miss_hook,
            cache_eviction_hook=self._eviction_hook,
            cache_remove_hook=self._remove_hook,
            cache_free_hook=self._free_hook,
            cache_name="FOOPointwise",
        )

        for idx in range(n_requests):
            if idx > 0 and idx % 1_000_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"    {idx:,}/{n_requests:,} ({elapsed:.1f}s), "
                    f"evictions={self.n_evictions:,}, "
                    f"decision_points={self.n_decision_points:,}, "
                    f"rows={self.n_rows_generated:,}",
                    flush=True,
                )

            obj_id = int(self.trace.obj_ids[idx])
            obj_size = int(self.trace.obj_sizes[idx])
            timestamp = int(self.trace.timestamps[idx])
            target_dvar = float(self.soft_dvars[idx])
            key = self._get_object_key(obj_id, obj_size)
            cache_obj_id = self._get_cache_obj_id(key)
            self.current_req_idx = idx

            req = lcs.Request()
            req.obj_id = cache_obj_id
            req.obj_size = obj_size
            req.clock_time = timestamp

            if obj_size <= self.cache_size:
                self.cache.get(req)
                self._cached_keys.add(key)
                self._lru_counter += 1
                self._lru[cache_obj_id] = self._lru_counter
                existing_state = self.object_states.get(key)
                if existing_state is not None and existing_state.n_accesses >= self.min_history_len:
                    self._eligible_cached_keys[key] = None

            if key not in self.object_states:
                self.object_states[key] = ObjectState(obj_id=obj_id, obj_size=obj_size)
            self.object_states[key].add_access(idx, target_dvar)
            if (
                key in self._cached_keys
                and self.object_states[key].n_accesses >= self.min_history_len
            ):
                self._eligible_cached_keys[key] = None

        elapsed = time.time() - start_time
        print(f"  Total evictions: {self.n_evictions:,}")
        print(f"  Decision points sampled: {self.n_decision_points:,}")
        print(f"  Total rows: {self.n_rows_generated:,}")
        print(f"  Total time: {elapsed:.1f}s")
        return self.n_rows_generated


def export_pointwise_libcachesim(
    trace: TraceData,
    foo_result: FOOResult,
    output_path: str,
    cache_size: int,
    max_samples_per_point: Optional[int] = None,
    min_history_len: int = 2,
    seed: int = 42,
) -> int:
    """Generate pointwise decision-point data using libCacheSim."""

    columns = [
        "obj_id",
        "obj_size",
        "mean_arr",
        "last_5_access_0",
        "last_5_access_1",
        "last_5_access_2",
        "last_5_access_3",
        "last_5_access_4",
        "now_last_space",
        "target_dvar",
    ]

    print(
        "  dvar summary: "
        f"mean={float(np.mean(foo_result.dvars)):.4f}, "
        f"p50={float(np.quantile(foo_result.dvars, 0.5)):.4f}, "
        f"p90={float(np.quantile(foo_result.dvars, 0.9)):.4f}"
    )

    generator = PointwiseGenerator(
        cache_size=cache_size,
        trace=trace,
        foo_result=foo_result,
        max_samples_per_eviction=max_samples_per_point,
        min_history_len=min_history_len,
        seed=seed,
    )

    with open(output_path, "w") as f:
        f.write(",".join(columns) + "\n")
        n_rows = generator.generate(f)

    if n_rows == 0:
        print("  WARNING: No pointwise rows generated!")
    else:
        print(f"  Saved to: {output_path}")
    return n_rows


__all__ = [
    "PointwiseGenerator",
    "export_pointwise_libcachesim",
    "load_dvars_from_cpp_foo",
]
