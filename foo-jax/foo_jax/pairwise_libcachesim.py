"""
Pairwise data generation using libCacheSim for accurate cache simulation.

Key improvements over manual simulation:
1. Accurate remaining space via get_occupied_byte()
2. Decision points at actual evictions (not fixed intervals)
3. Leverages battle-tested libCacheSim cache logic

Sampling strategies:
- random: Random sampling (baseline, may create trivially separable pairs)
- similar: Focus on pairs with similar features but different dvars (difficult cases)
- stratified: Mix of easy (30%) and difficult (70%) cases (recommended)

Output format (19 columns, matching original):
    hi_obj_id, hi_obj_size, hi_mean_arr, hi_last_5_access_0..4, hi_now_last_space,
    lo_obj_id, lo_obj_size, lo_mean_arr, lo_last_5_access_0..4, lo_now_last_space,
    label
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, IO
import numpy as np
import time

import libcachesim as lcs

from .trace_parser import TraceData
from .output import FOOResult


class SamplingStrategy(Enum):
    """Sampling strategy for pairwise data generation."""
    RANDOM = "random"           # Random sampling (baseline)
    SIMILAR = "similar"         # Focus on similar-feature pairs (difficult cases)
    STRATIFIED = "stratified"   # Mix of easy and difficult cases (recommended)


def load_dvars_from_cpp_foo(dvar_file: str, trace: TraceData) -> np.ndarray:
    """
    Load decision variables from C++ FOO output file.

    C++ FOO output format: timestamp id size dvar
    One line per request, ordered same as input trace.

    Args:
        dvar_file: Path to C++ FOO output file
        trace: Parsed trace data (for validation)

    Returns:
        Array of dvars aligned with trace requests
    """
    dvars = []
    with open(dvar_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                dvar = float(parts[3])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid dvar at line {line_no} in {dvar_file}: '{parts[3]}'"
                ) from exc
            if dvar < -1e-6 or dvar > 1.0 + 1e-6:
                raise ValueError(
                    f"Out-of-range dvar at line {line_no} in {dvar_file}: {dvar}"
                )
            dvars.append(dvar)

    dvars = np.array(dvars, dtype=np.float32)

    if len(dvars) != trace.n_requests:
        raise ValueError(
            f"Mismatch: dvar file has {len(dvars)} entries, "
            f"trace has {trace.n_requests} requests"
        )

    return dvars


@dataclass
class ObjectState:
    """Per-object state for feature computation."""
    obj_id: int
    obj_size: int
    access_vtimes: List[int] = field(default_factory=list)  # Recent vtimes (request indices)
    current_dvar: int = 0  # Most recent FOO decision (0 or 1)
    n_accesses: int = 0

    def add_access(self, vtime: int, dvar: int, max_history: int = 20):
        """Record an access and update dvar. vtime is the request index."""
        self.access_vtimes.append(vtime)
        if len(self.access_vtimes) > max_history:
            self.access_vtimes = self.access_vtimes[-max_history:]
        self.current_dvar = dvar
        self.n_accesses += 1

    def compute_mean_arr(self) -> float:
        """Compute mean inter-arrival time in vtime (request count)."""
        if len(self.access_vtimes) < 2:
            return -1.0
        intervals = [
            self.access_vtimes[i] - self.access_vtimes[i - 1]
            for i in range(1, len(self.access_vtimes))
        ]
        return float(np.mean(intervals))

    def compute_last_5_access(self, current_vtime: int) -> List[float]:
        """Compute vtime since last 1-5 accesses (in request count)."""
        result = [-1.0] * 5
        for i, vt in enumerate(reversed(self.access_vtimes[-5:])):
            if i < 5:
                result[i] = float(current_vtime - vt)
        return result


class PairwiseGenerator:
    """
    Generate pairwise training data at eviction decision points.

    Uses libCacheSim's PluginCache to:
    1. Simulate cache behavior following FOO decisions
    2. Sample pairs at actual eviction events (true decision points)
    3. Get accurate remaining space via get_occupied_byte()

    Sampling strategies:
    - random: Random sampling (baseline)
    - similar: Focus on pairs with similar features (difficult cases)
    - stratified: Mix of easy and difficult cases (recommended)
    """

    def __init__(
        self,
        cache_size: int,
        trace: TraceData,
        foo_result: FOOResult,
        max_pairs_per_eviction: int = 100,
        min_history_len: int = 2,
        seed: int = 42,
        sampling_strategy: str = "stratified",
    ):
        self.cache_size = cache_size
        self.trace = trace
        self.dvars = np.round(foo_result.dvars).astype(np.int32)
        self.max_pairs_per_eviction = max_pairs_per_eviction
        self.min_history_len = min_history_len
        self.seed = seed
        self.sampling_strategy = SamplingStrategy(sampling_strategy)

        # Object state tracking: (obj_id, obj_size) -> ObjectState
        self.object_states: Dict[Tuple[int, int], ObjectState] = {}

        # Incremental candidate indexing: avoid O(N) scan per eviction
        self._hi_keys: Set[Tuple[int, int]] = set()  # dvar=1, enough history
        self._lo_keys: Set[Tuple[int, int]] = set()  # dvar=0, enough history

        # Streaming output (set in generate())
        self._output_file: Optional[IO] = None
        self.n_pairs_generated: int = 0

        # Current request index (vtime for feature computation)
        self.current_req_idx = 0  # This IS the vtime

        # Statistics
        self.n_evictions = 0
        self.n_decision_points = 0

        # LRU state for eviction hook (stored in self for access from main loop)
        self._lru: Dict[int, int] = {}  # obj_id -> access counter
        self._lru_counter: int = 0
        self._cache_obj_ids: Dict[Tuple[int, int], int] = {}
        self._next_cache_obj_id: int = 1

        np.random.seed(seed)

    def _get_object_key(self, obj_id: int, obj_size: int) -> Tuple[int, int]:
        """Object key matching FOO's (id, size) semantics."""
        return (obj_id, obj_size)

    def _get_cache_obj_id(self, key: Tuple[int, int]) -> int:
        """
        Get a stable synthetic cache object ID for (obj_id, obj_size).

        libCacheSim cache identity is obj_id-only, while FOO semantics use
        (obj_id, obj_size). We map tuple keys to synthetic IDs to keep both
        models consistent.
        """
        cache_obj_id = self._cache_obj_ids.get(key)
        if cache_obj_id is not None:
            return cache_obj_id
        cache_obj_id = self._next_cache_obj_id
        self._cache_obj_ids[key] = cache_obj_id
        self._next_cache_obj_id += 1
        return cache_obj_id

    def _build_pair(
        self, hi_state: ObjectState, lo_state: ObjectState, remaining_space: float
    ) -> Tuple:
        """Build a pair tuple (19 columns) from hi and lo object states."""
        hi_mean_arr = hi_state.compute_mean_arr()
        hi_last_5 = hi_state.compute_last_5_access(self.current_req_idx)

        lo_mean_arr = lo_state.compute_mean_arr()
        lo_last_5 = lo_state.compute_last_5_access(self.current_req_idx)

        return (
            hi_state.obj_id,
            hi_state.obj_size,
            hi_mean_arr,
            hi_last_5[0],
            hi_last_5[1],
            hi_last_5[2],
            hi_last_5[3],
            hi_last_5[4],
            float(remaining_space),
            lo_state.obj_id,
            lo_state.obj_size,
            lo_mean_arr,
            lo_last_5[0],
            lo_last_5[1],
            lo_last_5[2],
            lo_last_5[3],
            lo_last_5[4],
            float(remaining_space),
            1,  # label
        )

    def _sample_random_pairs(
        self,
        hi_candidates: List[ObjectState],
        lo_candidates: List[ObjectState],
        remaining_space: float,
        max_pairs: int,
    ) -> List[Tuple]:
        """Random sampling (baseline strategy)."""
        pairs = []
        n_to_sample = min(max_pairs, len(hi_candidates) * len(lo_candidates))

        if n_to_sample > 0:
            hi_samples = np.random.choice(len(hi_candidates), size=n_to_sample, replace=True)
            lo_samples = np.random.choice(len(lo_candidates), size=n_to_sample, replace=True)

            for hi_idx, lo_idx in zip(hi_samples, lo_samples):
                pair = self._build_pair(
                    hi_candidates[hi_idx], lo_candidates[lo_idx], remaining_space
                )
                pairs.append(pair)

        return pairs

    def _sample_similar_pairs(
        self,
        hi_candidates: List[ObjectState],
        lo_candidates: List[ObjectState],
        remaining_space: float,
        max_pairs: int,
    ) -> List[Tuple]:
        """
        Sample pairs where features are SIMILAR but dvars differ.

        This captures "difficult cases" where FOO's decision differs from simple rules.
        Objects with similar mean_arr/last_access but different dvars reveal FOO's
        unique logic (e.g., object size considerations, global optimization).

        Performance optimization: Limits candidates to top-k by access count to avoid
        O(n*m) complexity with large candidate sets.
        """
        pairs = []

        # Limit candidates for performance (top by access count = most informative)
        MAX_CANDIDATES = 500
        if len(hi_candidates) > MAX_CANDIDATES:
            hi_candidates = sorted(hi_candidates, key=lambda s: s.n_accesses, reverse=True)[:MAX_CANDIDATES]
        if len(lo_candidates) > MAX_CANDIDATES:
            lo_candidates = sorted(lo_candidates, key=lambda s: s.n_accesses, reverse=True)[:MAX_CANDIDATES]

        # Compute features for all candidates
        hi_features = []
        for state in hi_candidates:
            mean_arr = state.compute_mean_arr()
            last_access = state.compute_last_5_access(self.current_req_idx)[0]
            if mean_arr >= 0 and last_access >= 0:
                hi_features.append((state, mean_arr, last_access, state.obj_size))

        lo_features = []
        for state in lo_candidates:
            mean_arr = state.compute_mean_arr()
            last_access = state.compute_last_5_access(self.current_req_idx)[0]
            if mean_arr >= 0 and last_access >= 0:
                lo_features.append((state, mean_arr, last_access, state.obj_size))

        if not hi_features or not lo_features:
            return pairs

        # For each hi, find lo with SIMILAR features
        for hi_state, hi_mean, hi_last, hi_size in hi_features:
            # Find similar lo candidates
            similar_lo = []
            for lo_state, lo_mean, lo_last, lo_size in lo_features:
                # Compute feature similarity (smaller = more similar)
                # Use relative difference to handle different scales
                mean_sim = abs(hi_mean - lo_mean) / max(hi_mean, lo_mean, 1.0)
                last_sim = abs(hi_last - lo_last) / max(hi_last, lo_last, 1.0)
                size_sim = abs(hi_size - lo_size) / max(hi_size, lo_size, 1.0)

                # Combined similarity score (lower = more similar)
                # Size weighted at 0.5 (less important than temporal features)
                similarity = mean_sim + last_sim + 0.5 * size_sim
                similar_lo.append((similarity, lo_state))

            if not similar_lo:
                continue

            # Sort by similarity (most similar first)
            similar_lo.sort(key=lambda x: x[0])

            # Sample from top-k most similar
            k = min(3, len(similar_lo))
            for _, lo_state in similar_lo[:k]:
                pair = self._build_pair(hi_state, lo_state, remaining_space)
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    return pairs

        return pairs

    def _sample_stratified_pairs(
        self,
        hi_candidates: List[ObjectState],
        lo_candidates: List[ObjectState],
        remaining_space: float,
        max_pairs: int,
    ) -> List[Tuple]:
        """
        Stratified sampling: mix of easy and difficult cases.

        - 70% difficult cases (similar features): Learn FOO's unique decisions
        - 30% easy cases (random): Provide baseline signal, prevent overfitting
        """
        # 70% difficult (similar features), 30% easy (random)
        n_difficult = int(max_pairs * 0.7)
        n_easy = max_pairs - n_difficult

        difficult_pairs = self._sample_similar_pairs(
            hi_candidates, lo_candidates, remaining_space, n_difficult
        )

        easy_pairs = self._sample_random_pairs(
            hi_candidates, lo_candidates, remaining_space, n_easy
        )

        return difficult_pairs + easy_pairs

    def _init_hook(self, params: Any) -> Dict:
        """Initialize cache state."""
        return {}

    def _hit_hook(self, data: Dict, req: lcs.Request) -> None:
        """Update LRU on hit."""
        self._lru_counter += 1
        self._lru[req.obj_id] = self._lru_counter

    def _miss_hook(self, data: Dict, req: lcs.Request) -> None:
        """Called on miss. Don't add to LRU yet — the object isn't admitted
        until after eviction completes. We add it in the main loop."""
        pass

    def _eviction_hook(self, data: Dict, req: lcs.Request) -> int:
        """
        Called when cache needs to evict.

        This is a TRUE DECISION POINT - sample pairs here using the configured strategy.
        Pairs are written directly to the output file (true streaming).
        """
        self.n_evictions += 1

        # Get remaining space BEFORE this eviction
        remaining_space = self.cache_size - self.cache.get_occupied_byte()

        # Build candidate lists from pre-indexed sets (O(|candidates|) not O(|all_objects|))
        hi_candidates = [self.object_states[k] for k in self._hi_keys]
        lo_candidates = [self.object_states[k] for k in self._lo_keys]

        # Sample pairs if we have both hi and lo candidates
        if hi_candidates and lo_candidates:
            self.n_decision_points += 1

            # Use strategy-based sampling
            if self.sampling_strategy == SamplingStrategy.SIMILAR:
                new_pairs = self._sample_similar_pairs(
                    hi_candidates, lo_candidates, remaining_space,
                    self.max_pairs_per_eviction
                )
            elif self.sampling_strategy == SamplingStrategy.STRATIFIED:
                new_pairs = self._sample_stratified_pairs(
                    hi_candidates, lo_candidates, remaining_space,
                    self.max_pairs_per_eviction
                )
            else:  # RANDOM
                new_pairs = self._sample_random_pairs(
                    hi_candidates, lo_candidates, remaining_space,
                    self.max_pairs_per_eviction
                )

            # Stream pairs directly to file
            if self._output_file is not None:
                for pair in new_pairs:
                    self._output_file.write(",".join(map(str, pair)) + "\n")
            self.n_pairs_generated += len(new_pairs)

        # Evict LRU item from self._lru
        if self._lru:
            lru_obj_id = min(self._lru, key=lambda k: self._lru[k])
            del self._lru[lru_obj_id]
            return lru_obj_id

        # Fallback: shouldn't happen
        return req.obj_id

    def _remove_hook(self, data: Dict, obj_id: int) -> None:
        """Called after eviction."""
        self._lru.pop(obj_id, None)

    def _free_hook(self, data: Dict) -> None:
        """Cleanup."""
        self._lru.clear()

    def generate(self, output_file=None) -> int:
        """
        Generate pairwise data by simulating cache with FOO decisions.

        Pairs are written directly to output_file during simulation (true streaming),
        avoiding memory accumulation for GB-scale traces.

        Args:
            output_file: File handle for streaming write (required for output)

        Returns:
            Number of pairs generated
        """
        start_time = time.time()
        self._output_file = output_file

        trace = self.trace
        n_requests = trace.n_requests

        print(f"  Processing {n_requests:,} requests...")

        # Create PluginCache
        # Note: We need to store self.cache for get_occupied_byte() in hooks
        self.cache = lcs.PluginCache(
            cache_size=self.cache_size,
            cache_init_hook=self._init_hook,
            cache_hit_hook=self._hit_hook,
            cache_miss_hook=self._miss_hook,
            cache_eviction_hook=self._eviction_hook,
            cache_remove_hook=self._remove_hook,
            cache_free_hook=self._free_hook,
            cache_name="FOOPairwise",
        )

        # Process trace
        for idx in range(n_requests):
            if idx > 0 and idx % 1_000_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"    {idx:,}/{n_requests:,} ({elapsed:.1f}s), "
                    f"evictions={self.n_evictions:,}, "
                    f"decision_points={self.n_decision_points:,}, "
                    f"pairs={self.n_pairs_generated:,}",
                    flush=True,
                )

            # Get request data
            obj_id = int(trace.obj_ids[idx])
            obj_size = int(trace.obj_sizes[idx])
            timestamp = int(trace.timestamps[idx])  # Only for libcachesim clock_time
            dvar = int(self.dvars[idx])
            key = self._get_object_key(obj_id, obj_size)
            cache_obj_id = self._get_cache_obj_id(key)

            # Update current vtime for feature computation in hooks
            self.current_req_idx = idx  # vtime = request index

            # Create request for libcachesim
            req = lcs.Request()
            req.obj_id = cache_obj_id
            req.obj_size = obj_size
            req.clock_time = timestamp

            # Process through cache FIRST (triggers hooks)
            # This way, eviction_hook sees state BEFORE current request
            # Only cache if dvar=1 AND object fits in cache
            if dvar == 1 and obj_size <= self.cache_size:
                self.cache.get(req)
                # Add to LRU AFTER admission (miss_hook doesn't add to avoid
                # evicting an object that isn't in the cache yet)
                self._lru_counter += 1
                self._lru[cache_obj_id] = self._lru_counter

            # Update object state AFTER cache processing
            # This ensures eviction decisions use pre-request state
            if key not in self.object_states:
                self.object_states[key] = ObjectState(obj_id=obj_id, obj_size=obj_size)
            self.object_states[key].add_access(idx, dvar)  # Use vtime (idx), not timestamp

            # Update incremental candidate index
            state = self.object_states[key]
            if state.n_accesses >= self.min_history_len:
                if state.current_dvar == 1:
                    self._lo_keys.discard(key)
                    self._hi_keys.add(key)
                else:
                    self._hi_keys.discard(key)
                    self._lo_keys.add(key)

        elapsed = time.time() - start_time
        print(f"  Total evictions: {self.n_evictions:,}")
        print(f"  Decision points sampled: {self.n_decision_points:,}")
        print(f"  Total pairs: {self.n_pairs_generated:,}")
        print(f"  Total time: {elapsed:.1f}s")

        return self.n_pairs_generated


def export_pairwise_libcachesim(
    trace: TraceData,
    foo_result: FOOResult,
    output_path: str,
    cache_size: int,
    max_pairs_per_point: int = 100,
    min_history_len: int = 2,
    seed: int = 42,
    sampling_strategy: str = "stratified",
) -> int:
    """
    Generate and export pairwise data using libCacheSim.

    Args:
        trace: Parsed trace data
        foo_result: FOO solver result with dvars
        output_path: Path to save CSV
        cache_size: Cache size in bytes
        max_pairs_per_point: Max pairs per decision point
        min_history_len: Minimum accesses for valid candidate
        seed: Random seed
        sampling_strategy: Sampling strategy ("random", "similar", "stratified")
            - random: Random sampling (baseline, may create trivially separable pairs)
            - similar: Focus on pairs with similar features (difficult cases)
            - stratified: Mix of easy (30%) and difficult (70%) cases (recommended)

    Returns:
        Number of pairs generated
    """
    # CSV header (original 19-column format)
    columns = [
        "hi_obj_id",
        "hi_obj_size",
        "hi_mean_arr",
        "hi_last_5_access_0",
        "hi_last_5_access_1",
        "hi_last_5_access_2",
        "hi_last_5_access_3",
        "hi_last_5_access_4",
        "hi_now_last_space",
        "lo_obj_id",
        "lo_obj_size",
        "lo_mean_arr",
        "lo_last_5_access_0",
        "lo_last_5_access_1",
        "lo_last_5_access_2",
        "lo_last_5_access_3",
        "lo_last_5_access_4",
        "lo_now_last_space",
        "label",
    ]

    # Print dvar distribution
    dvars = np.round(foo_result.dvars).astype(np.int32)
    n_dvar_0 = np.sum(dvars == 0)
    n_dvar_1 = np.sum(dvars == 1)
    print(
        f"  dvar distribution: hi={n_dvar_1:,} ({100*n_dvar_1/len(dvars):.1f}%), "
        f"lo={n_dvar_0:,} ({100*n_dvar_0/len(dvars):.1f}%)"
    )
    print(f"  Sampling strategy: {sampling_strategy}")

    # Create generator
    generator = PairwiseGenerator(
        cache_size=cache_size,
        trace=trace,
        foo_result=foo_result,
        max_pairs_per_eviction=max_pairs_per_point,
        min_history_len=min_history_len,
        seed=seed,
        sampling_strategy=sampling_strategy,
    )

    # Generate and write
    with open(output_path, "w") as f:
        # Write header
        f.write(",".join(columns) + "\n")

        # Generate pairs (writes to file in generate())
        n_pairs = generator.generate(output_file=f)

    if n_pairs == 0:
        print("  WARNING: No pairs generated!")
    else:
        print(f"  Saved to: {output_path}")

    return n_pairs
