"""Groupwise candidate export with soft or exact-victim supervision."""

from __future__ import annotations

from collections import OrderedDict
import json
import time
from pathlib import Path
from typing import Any, IO, Optional

import numpy as np

try:
    import libcachesim as lcs
except Exception:  # pragma: no cover - optional dependency for soft mode only
    lcs = None

from .output import FOOResult
from .pairwise_libcachesim import ObjectState, load_dvars_from_cpp_foo
from .trace_parser import TraceData


def _summarize_dvars(dvars: np.ndarray) -> tuple[np.ndarray, int]:
    dvars = np.asarray(dvars, dtype=np.float32)
    ambiguous = int(np.sum((dvars > 1e-6) & (dvars < 1.0 - 1e-6)))
    return (dvars > 0.5).astype(np.int8), ambiguous


class SoftGroupwiseGenerator:
    """Export cached eviction groups with dense soft labels."""

    def __init__(
        self,
        cache_size: int,
        trace: TraceData,
        foo_result: FOOResult,
        max_samples_per_eviction: Optional[int] = None,
        min_history_len: int = 2,
        seed: int = 42,
    ) -> None:
        if lcs is None:
            raise ImportError("libcachesim is required for soft groupwise export")

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
        self.n_groups_generated = 0
        self.n_rows_generated = 0
        self.n_uniform_groups_skipped = 0
        self._decision_id = 0

        self._lru: OrderedDict[int, None] = OrderedDict()
        self._cache_obj_ids: dict[tuple[int, int], int] = {}
        self._cache_obj_id_to_key: dict[int, tuple[int, int]] = {}
        self._cached_keys: set[tuple[int, int]] = set()
        self._eligible_cached_keys: set[tuple[int, int]] = set()

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
        decision_id: int,
        state: ObjectState,
    ) -> tuple[float | int, ...]:
        last_5 = state.compute_last_5_access(self.current_req_idx)
        current_target = float(state.current_dvar)
        next_target = self._next_access_target(state)
        recency = last_5[0]
        return (
            decision_id,
            state.obj_id,
            state.obj_size,
            state.compute_mean_arr(),
            last_5[0],
            last_5[1],
            last_5[2],
            last_5[3],
            last_5[4],
            recency,
            current_target,
            next_target,
        )

    def _sample_candidate_keys(
        self,
        candidate_keys: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
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

    @staticmethod
    def _has_group_signal(values: list[float]) -> bool:
        if len(values) < 2:
            return False
        arr = np.asarray(values, dtype=np.float32)
        return bool(np.ptp(arr) > 1e-6)

    def _init_hook(self, params: Any) -> dict[str, Any]:
        return {}

    def _hit_hook(self, data: dict[str, Any], req: Any) -> None:
        return None

    def _miss_hook(self, data: dict[str, Any], req: Any) -> None:
        return None

    def _eviction_hook(self, data: dict[str, Any], req: Any) -> int:
        self.n_evictions += 1

        candidate_keys = self._sample_candidate_keys(list(self._eligible_cached_keys))
        if candidate_keys and self._output_file is not None:
            self.n_decision_points += 1

            rows: list[tuple[float | int, ...]] = []
            current_targets: list[float] = []
            next_targets: list[float] = []
            decision_id = self._decision_id

            for key in candidate_keys:
                state = self.object_states[key]
                row = self._build_row(decision_id, state)
                rows.append(row)
                current_targets.append(float(row[-2]))
                next_targets.append(float(row[-1]))

            if self._has_group_signal(current_targets) or self._has_group_signal(next_targets):
                self._output_file.write(
                    "\n".join(",".join(map(str, row)) for row in rows) + "\n"
                )
                self.n_groups_generated += 1
                self.n_rows_generated += len(rows)
                self._decision_id += 1
            else:
                self.n_uniform_groups_skipped += 1

        if self._lru:
            lru_obj_id, _ = self._lru.popitem(last=False)
            return lru_obj_id

        return req.obj_id

    def _remove_hook(self, data: dict[str, Any], obj_id: int) -> None:
        self._lru.pop(obj_id, None)
        key = self._cache_obj_id_to_key.get(obj_id)
        if key is not None:
            self._cached_keys.discard(key)
            self._eligible_cached_keys.discard(key)

    def _free_hook(self, data: dict[str, Any]) -> None:
        self._lru.clear()
        self._cached_keys.clear()
        self._eligible_cached_keys.clear()

    def generate(self, output_file: IO[str]) -> tuple[int, int]:
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
            cache_name="FOOGroupwiseSoft",
        )

        for idx in range(n_requests):
            if idx > 0 and idx % 1_000_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"    {idx:,}/{n_requests:,} ({elapsed:.1f}s), "
                    f"evictions={self.n_evictions:,}, "
                    f"decision_points={self.n_decision_points:,}, "
                    f"groups={self.n_groups_generated:,}, "
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
                self._lru.pop(cache_obj_id, None)
                self._lru[cache_obj_id] = None
                existing_state = self.object_states.get(key)
                if existing_state is not None and existing_state.n_accesses >= self.min_history_len:
                    self._eligible_cached_keys.add(key)

            if key not in self.object_states:
                self.object_states[key] = ObjectState(obj_id=obj_id, obj_size=obj_size)
            self.object_states[key].add_access(idx, target_dvar)
            if (
                key in self._cached_keys
                and self.object_states[key].n_accesses >= self.min_history_len
            ):
                self._eligible_cached_keys.add(key)

        elapsed = time.time() - start_time
        print(f"  Total evictions: {self.n_evictions:,}")
        print(f"  Decision points visited: {self.n_decision_points:,}")
        print(f"  Informative groups: {self.n_groups_generated:,}")
        print(f"  Uniform groups skipped: {self.n_uniform_groups_skipped:,}")
        print(f"  Total rows: {self.n_rows_generated:,}")
        print(f"  Total time: {elapsed:.1f}s")
        return self.n_groups_generated, self.n_rows_generated

    def meta(self) -> dict[str, int | float | None]:
        return {
            "n_evictions": int(self.n_evictions),
            "n_decision_points": int(self.n_decision_points),
            "n_uniform_groups_skipped": int(self.n_uniform_groups_skipped),
        }


class ExactVictimGroupwiseGenerator:
    """Export sharp keep/victim labels on LRU residents using a feasible projected FOO keep-set."""

    def __init__(
        self,
        cache_size: int,
        trace: TraceData,
        foo_result: FOOResult,
        max_samples_per_eviction: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.cache_size = cache_size
        self.trace = trace
        self.raw_dvars = np.asarray(foo_result.dvars, dtype=np.float32)
        _, self.n_ambiguous_dvars = _summarize_dvars(foo_result.dvars)
        self.max_samples_per_eviction = max_samples_per_eviction
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._output_file: Optional[IO[str]] = None

        self.object_states: dict[tuple[int, int], ObjectState] = {}
        self._exact_active_keys: dict[tuple[int, int], tuple[int, float]] = {}
        self._exact_occupied_bytes = 0
        self._current_exact_post_keys: set[tuple[int, int]] = set()
        self._current_projected_keep = False
        self._decision_id = 0

        self._physical_cache: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._physical_occupied_bytes = 0

        self.current_req_idx = 0
        self.n_exact_hits = 0
        self.n_exact_misses = 0
        self.n_exact_retained_misses = 0
        self.n_exact_rejected_misses = 0
        self.n_physical_hits = 0
        self.n_physical_misses = 0
        self.n_oversized_requests = 0
        self.n_eviction_decisions = 0
        self.n_groups_generated = 0
        self.n_rows_generated = 0
        self.n_uniform_groups_skipped = 0
        self.n_truncated_groups = 0
        self.n_required_survivor_overrides = 0
        self.n_projection_drops = 0
        self.n_bytes_evicted_total = 0
        self.n_victim_rows_total = 0
        self.n_survivor_rows_total = 0

    def _get_object_key(self, obj_id: int, obj_size: int) -> tuple[int, int]:
        return (obj_id, obj_size)

    def _build_row(
        self,
        decision_id: int,
        state: ObjectState,
        keep_label: int,
    ) -> tuple[float | int, ...]:
        last_5 = state.compute_last_5_access(self.current_req_idx)
        recency = last_5[0]
        victim_label = 1 - keep_label
        return (
            decision_id,
            state.obj_id,
            state.obj_size,
            state.compute_mean_arr(),
            last_5[0],
            last_5[1],
            last_5[2],
            last_5[3],
            last_5[4],
            recency,
            state.n_accesses,
            keep_label,
            victim_label,
        )

    def _sample_candidate_keys(
        self,
        candidate_keys: set[tuple[int, int]],
        victim_keys: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        victims = list(victim_keys)
        survivors = list(candidate_keys - victim_keys)
        if not victims or not survivors:
            return []

        if (
            self.max_samples_per_eviction is None
            or len(candidate_keys) <= self.max_samples_per_eviction
        ):
            return victims + survivors

        survivor_budget = self.max_samples_per_eviction - len(victims)
        if survivor_budget <= 0:
            survivor_budget = 1
            self.n_required_survivor_overrides += 1

        if survivor_budget >= len(survivors):
            return victims + survivors

        self.n_truncated_groups += 1
        sample_idx = self._rng.choice(
            len(survivors),
            size=survivor_budget,
            replace=False,
        )
        sampled_survivors = [survivors[int(idx)] for idx in sample_idx]
        return victims + sampled_survivors

    def _emit_group(
        self,
        *,
        candidate_keys: set[tuple[int, int]],
        victim_keys: set[tuple[int, int]],
    ) -> None:
        sampled_candidate_keys = self._sample_candidate_keys(candidate_keys, victim_keys)
        if not sampled_candidate_keys:
            self.n_uniform_groups_skipped += 1
            return

        decision_id = self._decision_id
        rows: list[tuple[float | int, ...]] = []
        keep_count = 0
        victim_count = 0

        for key in sampled_candidate_keys:
            keep_label = 0 if key in victim_keys else 1
            if keep_label:
                keep_count += 1
            else:
                victim_count += 1
            rows.append(
                self._build_row(
                    decision_id,
                    self.object_states[key],
                    keep_label=keep_label,
                )
            )

        if keep_count == 0 or victim_count == 0:
            self.n_uniform_groups_skipped += 1
            return

        self._output_file.write("\n".join(",".join(map(str, row)) for row in rows) + "\n")
        self.n_groups_generated += 1
        self.n_rows_generated += len(rows)
        self.n_victim_rows_total += victim_count
        self.n_survivor_rows_total += keep_count
        self._decision_id += 1

    def _prepare_exact_post_state(self, key: tuple[int, int], obj_size: int) -> None:
        active_interval = self._exact_active_keys.get(key)
        if active_interval is not None:
            expected_end_idx, _ = active_interval
            if expected_end_idx != self.current_req_idx:
                raise RuntimeError(
                    f"FOO interval inconsistency for key={key}: expected next access "
                    f"{expected_end_idx}, observed {self.current_req_idx}"
                )
            self.n_exact_hits += 1
            del self._exact_active_keys[key]
            self._exact_occupied_bytes -= obj_size
        else:
            self.n_exact_misses += 1

        next_idx = int(self.trace.next_access_idx[self.current_req_idx])
        raw_keep_score = float(self.raw_dvars[self.current_req_idx])
        current_added = False
        if next_idx >= 0 and raw_keep_score > 1e-6 and obj_size <= self.cache_size:
            self._exact_active_keys[key] = (next_idx, raw_keep_score)
            self._exact_occupied_bytes += obj_size
            current_added = True

        while self._exact_occupied_bytes > self.cache_size and self._exact_active_keys:
            victim_key, (victim_end_idx, victim_score) = min(
                self._exact_active_keys.items(),
                key=lambda item: (
                    item[1][1],
                    -item[1][0],
                    -item[0][1],
                    item[0][0],
                ),
            )
            del self._exact_active_keys[victim_key]
            self._exact_occupied_bytes -= victim_key[1]
            self.n_projection_drops += 1
            if victim_key == key:
                current_added = False

        if active_interval is None:
            if current_added:
                self.n_exact_retained_misses += 1
            else:
                self.n_exact_rejected_misses += 1

        if self._exact_occupied_bytes > self.cache_size:
            raise RuntimeError(
                f"Reconstructed exact FOO keep-set exceeds capacity at request {self.current_req_idx}: "
                f"{self._exact_occupied_bytes} > {self.cache_size}"
            )
        self._current_exact_post_keys = set(self._exact_active_keys)
        self._current_projected_keep = current_added

    def _simulate_physical_lru(self, key: tuple[int, int], obj_size: int) -> None:
        if obj_size > self.cache_size:
            return

        if key in self._physical_cache:
            self.n_physical_hits += 1
            self._physical_cache.move_to_end(key, last=True)
            return

        self.n_physical_misses += 1
        while self._physical_occupied_bytes + obj_size > self.cache_size and self._physical_cache:
            self.n_eviction_decisions += 1
            candidate_keys = set(self._physical_cache.keys())
            victim_keys = candidate_keys - self._current_exact_post_keys
            self.n_bytes_evicted_total += int(sum(item[1] for item in victim_keys))
            self._emit_group(
                candidate_keys=candidate_keys,
                victim_keys=victim_keys,
            )
            victim_key, _ = self._physical_cache.popitem(last=False)
            self._physical_occupied_bytes -= victim_key[1]

        self._physical_cache[key] = None
        self._physical_occupied_bytes += obj_size

    def generate(self, output_file: IO[str]) -> tuple[int, int]:
        self._output_file = output_file
        start_time = time.time()
        n_requests = self.trace.n_requests
        progress_interval = max(10_000, min(1_000_000, n_requests // 10 or 1))

        print(f"  Processing {n_requests:,} requests...")
        if self.n_ambiguous_dvars > 0:
            print(
                f"  Warning: projected {self.n_ambiguous_dvars:,} non-binary dvars into a feasible keep-set"
            )

        for idx in range(n_requests):
            if idx > 0 and idx % progress_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"    {idx:,}/{n_requests:,} ({elapsed:.1f}s), "
                    f"exact_hits={self.n_exact_hits:,}, exact_misses={self.n_exact_misses:,}, "
                    f"physical_hits={self.n_physical_hits:,}, physical_misses={self.n_physical_misses:,}, "
                    f"evictions={self.n_eviction_decisions:,}, groups={self.n_groups_generated:,}, "
                    f"rows={self.n_rows_generated:,}",
                    flush=True,
                )

            obj_id = int(self.trace.obj_ids[idx])
            obj_size = int(self.trace.obj_sizes[idx])
            key = self._get_object_key(obj_id, obj_size)
            self.current_req_idx = idx

            if obj_size > self.cache_size:
                self.n_oversized_requests += 1

            self._prepare_exact_post_state(key, obj_size)
            self._simulate_physical_lru(key, obj_size)

            state = self.object_states.get(key)
            if state is None:
                state = ObjectState(obj_id=obj_id, obj_size=obj_size)
                self.object_states[key] = state
            state.add_access(idx, int(self._current_projected_keep))

        elapsed = time.time() - start_time
        print(f"  Exact hits reconstructed: {self.n_exact_hits:,}")
        print(f"  Exact misses reconstructed: {self.n_exact_misses:,}")
        print(f"  Exact retained misses: {self.n_exact_retained_misses:,}")
        print(f"  Exact rejected misses skipped: {self.n_exact_rejected_misses:,}")
        print(f"  Physical LRU hits: {self.n_physical_hits:,}")
        print(f"  Physical LRU misses: {self.n_physical_misses:,}")
        print(f"  Oversized requests skipped: {self.n_oversized_requests:,}")
        print(f"  Eviction decisions: {self.n_eviction_decisions:,}")
        print(f"  Groups generated: {self.n_groups_generated:,}")
        print(f"  Rows generated: {self.n_rows_generated:,}")
        print(f"  Victim rows: {self.n_victim_rows_total:,}")
        print(f"  Survivor rows: {self.n_survivor_rows_total:,}")
        print(f"  Uniform groups skipped: {self.n_uniform_groups_skipped:,}")
        print(f"  Truncated groups: {self.n_truncated_groups:,}")
        print(f"  Survivor-budget overrides: {self.n_required_survivor_overrides:,}")
        print(f"  Capacity-projection drops: {self.n_projection_drops:,}")
        print(f"  Total bytes labeled victim across emitted decisions: {self.n_bytes_evicted_total:,}")
        print(f"  Total time: {elapsed:.1f}s")
        return self.n_groups_generated, self.n_rows_generated

    def meta(self) -> dict[str, int | float | None]:
        return {
            "n_exact_hits": int(self.n_exact_hits),
            "n_exact_misses": int(self.n_exact_misses),
            "n_exact_retained_misses": int(self.n_exact_retained_misses),
            "n_exact_rejected_misses": int(self.n_exact_rejected_misses),
            "n_physical_hits": int(self.n_physical_hits),
            "n_physical_misses": int(self.n_physical_misses),
            "n_oversized_requests": int(self.n_oversized_requests),
            "n_eviction_decisions": int(self.n_eviction_decisions),
            "n_uniform_groups_skipped": int(self.n_uniform_groups_skipped),
            "n_truncated_groups": int(self.n_truncated_groups),
            "n_required_survivor_overrides": int(self.n_required_survivor_overrides),
            "n_projection_drops": int(self.n_projection_drops),
            "victim_rows_generated": int(self.n_victim_rows_total),
            "survivor_rows_generated": int(self.n_survivor_rows_total),
            "victim_labeled_bytes_total": int(self.n_bytes_evicted_total),
            "ambiguous_dvars_rounded": int(self.n_ambiguous_dvars),
        }


def export_groupwise_libcachesim(
    trace: TraceData,
    foo_result: FOOResult,
    output_path: str,
    cache_size: int,
    max_samples_per_point: Optional[int] = None,
    min_history_len: int = 2,
    seed: int = 42,
    target_mode: str = "soft",
) -> tuple[int, int]:
    """Generate groupwise data with dense soft or exact-victim labels."""

    if target_mode == "soft":
        columns = [
            "decision_id",
            "obj_id",
            "obj_size",
            "mean_arr",
            "last_5_access_0",
            "last_5_access_1",
            "last_5_access_2",
            "last_5_access_3",
            "last_5_access_4",
            "now_last_space",
            "current_dvar",
            "next_dvar",
        ]
        generator: SoftGroupwiseGenerator | ExactVictimGroupwiseGenerator = SoftGroupwiseGenerator(
            cache_size=cache_size,
            trace=trace,
            foo_result=foo_result,
            max_samples_per_eviction=max_samples_per_point,
            min_history_len=min_history_len,
            seed=seed,
        )
    elif target_mode == "exact-victim":
        columns = [
            "decision_id",
            "obj_id",
            "obj_size",
            "mean_arr",
            "last_5_access_0",
            "last_5_access_1",
            "last_5_access_2",
            "last_5_access_3",
            "last_5_access_4",
            "now_last_space",
            "access_count",
            "keep_label",
            "victim_label",
        ]
        generator = ExactVictimGroupwiseGenerator(
            cache_size=cache_size,
            trace=trace,
            foo_result=foo_result,
            max_samples_per_eviction=max_samples_per_point,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported groupwise target_mode={target_mode}")

    print(
        "  dvar summary: "
        f"mean={float(np.mean(foo_result.dvars)):.4f}, "
        f"p50={float(np.quantile(foo_result.dvars, 0.5)):.4f}, "
        f"p90={float(np.quantile(foo_result.dvars, 0.9)):.4f}"
    )

    output_file_path = Path(output_path)
    with output_file_path.open("w") as f:
        f.write(",".join(columns) + "\n")
        n_groups, n_rows = generator.generate(f)

    meta_path = output_file_path.with_suffix(output_file_path.suffix + ".meta.json")
    meta_payload = {
        "target_mode": target_mode,
        "groups_generated": int(n_groups),
        "rows_generated": int(n_rows),
        "cache_size": int(cache_size),
        "n_requests": int(trace.n_requests),
        "n_unique_objects": int(trace.n_unique_objects),
        "max_samples_per_point": (
            None if max_samples_per_point is None else int(max_samples_per_point)
        ),
        "min_history_len": int(min_history_len),
        "seed": int(seed),
        "available_numeric_cols": columns[2:-2] if target_mode == "exact-victim" else columns[2:-2],
    }
    meta_payload.update(generator.meta())
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n")

    if n_rows == 0:
        print("  WARNING: No groupwise rows generated!")
    else:
        print(f"  Saved to: {output_path}")
        print(f"  Sidecar metadata: {meta_path}")
    return n_groups, n_rows


__all__ = [
    "SoftGroupwiseGenerator",
    "ExactVictimGroupwiseGenerator",
    "export_groupwise_libcachesim",
    "load_dvars_from_cpp_foo",
]
