"""
OracleGeneral trace format parser with zstd compression support.

Binary format (24 bytes per record, little-endian):
    - uint32 timestamp        (4 bytes)
    - uint64 obj_id           (8 bytes)
    - uint32 obj_size         (4 bytes)
    - int64  next_access_vtime (8 bytes, -1 if no next access)

Reference: /lib/trace/oracle_general_reader.h
"""

from dataclasses import dataclass
from typing import Optional
import struct
import numpy as np

# Try to import Rust fast parser
try:
    import foo_jax_fast_parser as _rust_parser
    _HAS_RUST_PARSER = hasattr(_rust_parser, "parse_trace_fast")
except ImportError:
    _HAS_RUST_PARSER = False


@dataclass
class TraceData:
    """
    Parsed trace data with precomputed topology arrays.

    Attributes:
        timestamps: Request timestamps (uint32)
        obj_ids: Object IDs (uint64)
        obj_sizes: Object sizes in bytes (uint32)
        next_access_idx: Index of next access for same object, -1 if none (int32)
        prev_access_idx: Index of previous access for same object, -1 if none (int32)
        n_requests: Total number of requests
        n_unique_objects: Number of unique (obj_id, obj_size) pairs
    """
    timestamps: np.ndarray      # uint32[N]
    obj_ids: np.ndarray         # uint64[N]
    obj_sizes: np.ndarray       # uint32[N]
    next_access_idx: np.ndarray # int32[N]
    prev_access_idx: np.ndarray # int32[N]
    n_requests: int
    n_unique_objects: int


def parse_trace(
    path: str,
    max_requests: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    use_rust: bool = True
) -> TraceData:
    """
    Parse OracleGeneral binary trace file.

    Supports both uncompressed (.dat) and zstd-compressed (.zst) files.
    Uses Rust extension for ~20-50x faster parsing when available.

    Args:
        path: Path to trace file
        max_requests: Maximum number of requests to parse (None for all)
        progress_callback: Optional callback for progress reporting
        use_rust: Use Rust extension if available (default True)

    Returns:
        TraceData with parsed records and precomputed topology
    """
    # Use Rust parser if available and requested
    if use_rust and _HAS_RUST_PARSER:
        return _parse_trace_rust(path, max_requests, progress_callback)

    # Fall back to Python implementation
    return _parse_trace_python(path, max_requests, progress_callback)


def _parse_trace_rust(
    path: str,
    max_requests: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> TraceData:
    """Parse trace using Rust extension."""
    if progress_callback:
        progress_callback("Parsing trace with Rust extension...")

    try:
        result = _rust_parser.parse_trace_fast(path, max_requests)
    except Exception as exc:
        # Fall back to Python parser if the extension is present but broken or
        # compiled against an incompatible ABI.
        if progress_callback:
            progress_callback(f"Rust parser unavailable ({exc}), falling back to Python parser")
        return _parse_trace_python(path, max_requests, progress_callback)

    return TraceData(
        timestamps=np.asarray(result.get_timestamps()),
        obj_ids=np.asarray(result.get_obj_ids()),
        obj_sizes=np.asarray(result.get_obj_sizes()),
        next_access_idx=np.asarray(result.get_next_access_idx()),
        prev_access_idx=np.asarray(result.get_prev_access_idx()),
        n_requests=result.n_requests,
        n_unique_objects=result.n_unique_objects
    )


def _parse_trace_python(
    path: str,
    max_requests: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> TraceData:
    """Parse trace using pure Python with numpy vectorization."""
    import zstandard as zstd

    # Read and decompress file
    is_zst = path.endswith('.zst')
    record_size = 24  # 4 + 8 + 4 + 8 bytes

    if progress_callback:
        progress_callback(f"Reading {'compressed ' if is_zst else ''}trace file...")

    print(f"  Loading trace: {path}", flush=True)

    if is_zst:
        # For zst files, only decompress what we need
        dctx = zstd.ZstdDecompressor()
        with open(path, 'rb') as f:
            if max_requests is not None:
                # Read only needed bytes + some buffer
                needed_bytes = max_requests * record_size
                print(f"  Streaming decompression (need {needed_bytes/1024/1024:.1f} MB)...", flush=True)
                with dctx.stream_reader(f) as reader:
                    data = reader.read(needed_bytes)
            else:
                # Need full file
                print(f"  Full decompression...", flush=True)
                with dctx.stream_reader(f) as reader:
                    data = reader.read()
    else:
        with open(path, 'rb') as f:
            if max_requests is not None:
                data = f.read(max_requests * record_size)
            else:
                data = f.read()

    print(f"  Read size: {len(data)/1024/1024:.1f} MB", flush=True)

    # Validate and parse records
    record_size = 24  # 4 + 8 + 4 + 8 bytes
    n_records = len(data) // record_size

    if len(data) % record_size != 0:
        raise ValueError(f"Invalid trace file: size {len(data)} is not multiple of {record_size}")

    if max_requests is not None and max_requests < n_records:
        n_records = max_requests
        # Truncate data to avoid unnecessary processing
        data = data[:n_records * record_size]

    print(f"  Parsing {n_records:,} records...", flush=True)

    if progress_callback:
        progress_callback(f"Parsing {n_records:,} records...")

    # FAST: Use numpy structured array for vectorized parsing
    dtype = np.dtype([
        ('timestamp', '<u4'),   # uint32 little-endian
        ('obj_id', '<u8'),      # uint64 little-endian
        ('obj_size', '<u4'),    # uint32 little-endian
        ('next_vtime', '<i8')   # int64 little-endian (unused)
    ])

    records = np.frombuffer(data, dtype=dtype, count=n_records)

    # Extract arrays (these are views, very fast)
    timestamps = records['timestamp'].copy()
    obj_ids = records['obj_id'].copy()
    obj_sizes = records['obj_size'].copy()

    # Filter zero-size objects
    valid_mask = obj_sizes > 0
    if not np.all(valid_mask):
        timestamps = timestamps[valid_mask]
        obj_ids = obj_ids[valid_mask]
        obj_sizes = obj_sizes[valid_mask]
        print(f"  Filtered {np.sum(~valid_mask):,} zero-size objects", flush=True)

    n_requests = len(timestamps)

    print(f"  Building topology for {n_requests:,} requests...", flush=True)
    if progress_callback:
        progress_callback(f"Building topology for {n_requests:,} requests...")

    # Build topology: compute next_access_idx and prev_access_idx
    next_access_idx, prev_access_idx, n_unique = _build_access_topology(
        obj_ids, obj_sizes
    )

    return TraceData(
        timestamps=timestamps,
        obj_ids=obj_ids,
        obj_sizes=obj_sizes,
        next_access_idx=next_access_idx,
        prev_access_idx=prev_access_idx,
        n_requests=n_requests,
        n_unique_objects=n_unique
    )


def _build_access_topology(
    obj_ids: np.ndarray,
    obj_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build access topology arrays mapping each request to its prev/next occurrence.

    Uses (obj_id, obj_size) as key to match C++ behavior where objects with same ID
    but different sizes are treated as different objects.

    Args:
        obj_ids: Object ID array
        obj_sizes: Object size array

    Returns:
        (next_access_idx, prev_access_idx, n_unique_objects)
    """
    n = len(obj_ids)
    next_access_idx = np.full(n, -1, dtype=np.int32)
    prev_access_idx = np.full(n, -1, dtype=np.int32)

    # Map (obj_id, obj_size) -> last seen index
    # Using tuple as dict key for (id, size) pair
    last_seen: dict[tuple[int, int], int] = {}

    for i in range(n):
        key = (int(obj_ids[i]), int(obj_sizes[i]))

        if key in last_seen:
            prev_idx = last_seen[key]
            prev_access_idx[i] = prev_idx
            next_access_idx[prev_idx] = i

        last_seen[key] = i

    n_unique = len(last_seen)

    return next_access_idx, prev_access_idx, n_unique


def parse_trace_fast(
    path: str,
    max_requests: Optional[int] = None
) -> TraceData:
    """
    Fast trace parser using numpy for vectorized parsing.

    Approximately 5-10x faster than parse_trace() for large files.
    """
    import zstandard as zstd

    # Read and decompress file
    is_zst = path.endswith('.zst')

    with open(path, 'rb') as f:
        if is_zst:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = reader.read()
        else:
            data = f.read()

    # Validate
    record_size = 24
    n_records = len(data) // record_size

    if max_requests is not None and max_requests < n_records:
        n_records = max_requests
        data = data[:n_records * record_size]

    # Parse using numpy structured array (vectorized)
    dtype = np.dtype([
        ('timestamp', '<u4'),    # uint32 little-endian
        ('obj_id', '<u8'),       # uint64 little-endian
        ('obj_size', '<u4'),     # uint32 little-endian
        ('next_vtime', '<i8'),   # int64 little-endian
    ])

    records = np.frombuffer(data, dtype=dtype, count=n_records)

    # Filter out zero-size objects
    valid_mask = records['obj_size'] > 0
    records = records[valid_mask]

    timestamps = records['timestamp'].copy()
    obj_ids = records['obj_id'].copy()
    obj_sizes = records['obj_size'].copy()

    n_requests = len(timestamps)

    # Build topology
    next_access_idx, prev_access_idx, n_unique = _build_access_topology(
        obj_ids, obj_sizes
    )

    return TraceData(
        timestamps=timestamps,
        obj_ids=obj_ids,
        obj_sizes=obj_sizes,
        next_access_idx=next_access_idx,
        prev_access_idx=prev_access_idx,
        n_requests=n_requests,
        n_unique_objects=n_unique
    )


# Convenience function for loading test traces
def load_test_trace(name: str = "tiny") -> TraceData:
    """
    Load a test trace from the cuopt-python directory.

    Args:
        name: One of "tiny" (6 records), "100", "10k", "100k"
    """
    import os
    base_path = "/home/ubuntu/ssd/optimalwebcaching/cuopt-python"

    traces = {
        "tiny": "test_trace_tiny.dat",
        "100": "test_trace_100.dat",
        "10k": "test_trace_10k.dat",
        "100k": "test_trace_100k.dat",
    }

    if name not in traces:
        raise ValueError(f"Unknown test trace: {name}. Available: {list(traces.keys())}")

    path = os.path.join(base_path, traces[name])
    return parse_trace_fast(path)
