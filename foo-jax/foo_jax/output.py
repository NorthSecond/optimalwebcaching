"""
Output extraction for FOO algorithm.

Converts flow solution to decision variables and computes hit ratio metrics.

Decision variable interpretation:
    dvar = (size - flow) / size
    - dvar = 1.0 -> object completely cached (HIT)
    - dvar = 0.0 -> object completely evicted (MISS)
    - 0 < dvar < 1 -> fractional solution (LP relaxation)

Reference: /OHRgoal/FOO/foo.cpp:52-69
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import jax.numpy as jnp

from .trace_parser import TraceData
from .topology import Topology
from .solver import SolverResult


@dataclass
class FOOResult:
    """Complete result from FOO algorithm."""
    # Decision variables per request
    dvars: np.ndarray  # float32[n_requests], 0=MISS, 1=HIT

    # Metrics
    n_requests: int
    n_unique_objects: int
    float_hits: float  # Sum of dvars (fractional hits)
    integer_hits: int  # Count of dvars > 0.99 (near-certain hits)
    ohr: float  # Object Hit Ratio = hits / requests

    # Solver info
    primal_obj: float
    iterations: int
    converged: bool

    # Raw data for further analysis
    x_outer: np.ndarray  # Flow on outer arcs


def extract_decision_variables(
    trace: TraceData,
    topo: Topology,
    x_outer: jnp.ndarray
) -> np.ndarray:
    """
    Compute per-request decision variables from flow solution.

    The FOO algorithm assigns decision variables to the FIRST occurrence
    of each object interval. Second occurrences (actual hits) get dvar=0.

    Formula: dvar = (size - flow) / size
        - If flow = 0: object was completely cached -> dvar = 1 (will HIT)
        - If flow = size: object was evicted -> dvar = 0 (will MISS)

    C++ reference: /OHRgoal/FOO/foo.cpp:52-69

    Args:
        trace: Original trace data
        topo: Network topology
        x_outer: Flow on outer arcs from solver

    Returns:
        Array of dvars, one per request
    """
    n_requests = trace.n_requests
    dvars = np.zeros(n_requests, dtype=np.float32)

    # Outer arcs represent object intervals
    # outer_trace_idx points to the FIRST occurrence (where arc starts)
    # dvar = (size - flow) / size

    x_outer_np = np.asarray(x_outer)
    capacity_np = np.asarray(topo.outer_capacity)
    trace_idx_np = np.asarray(topo.outer_trace_idx)

    # Compute dvars for first occurrences (those with outer arcs)
    arc_dvars = (capacity_np - x_outer_np) / capacity_np

    # Scatter to trace positions
    # Note: trace_idx_np contains the FIRST occurrence index for each arc
    for i in range(len(trace_idx_np)):
        dvars[trace_idx_np[i]] = arc_dvars[i]

    # Second occurrences (those without outer arcs, i.e., hits) remain 0
    # This matches C++ logic where arcId == -1 gives dvar = 0

    return dvars


def compute_metrics(
    trace: TraceData,
    dvars: np.ndarray,
    primal_obj: float
) -> dict:
    """
    Compute OHR and related metrics.

    C++ formula:
        hitc = reqc - uniqc - solval
        OHR = hitc / reqc

    Where:
        reqc = total requests
        uniqc = unique objects (first occurrences are always misses)
        solval = solver's primal objective (sum of fractional evictions)

    Args:
        trace: Original trace data
        dvars: Decision variables per request
        primal_obj: Solver's primal objective value

    Returns:
        Dictionary with metrics
    """
    n_requests = trace.n_requests
    n_unique = trace.n_unique_objects

    # Float hits: sum of dvars (fractional)
    float_hits = float(np.sum(dvars))

    # Integer hits: count near-1 dvars (confident hits)
    integer_hits = int(np.sum(dvars > 0.99))

    # OHR calculation (matching C++ output)
    # hitc = reqc - uniqc - solval
    # Note: primal_obj represents sum(cost * flow) = sum((1/size) * evicted_bytes)
    # This equals the number of fractional misses
    hitc = n_requests - n_unique - primal_obj
    ohr = hitc / n_requests if n_requests > 0 else 0.0

    return {
        'n_requests': n_requests,
        'n_unique_objects': n_unique,
        'float_hits': float_hits,
        'integer_hits': integer_hits,
        'hitc': hitc,
        'ohr': ohr,
        'primal_obj': primal_obj
    }


def process_result(
    trace: TraceData,
    topo: Topology,
    solver_result: SolverResult
) -> FOOResult:
    """
    Process solver result into complete FOO output.

    Args:
        trace: Original trace data
        topo: Network topology
        solver_result: Result from r2HPDHG solver

    Returns:
        Complete FOO result with dvars and metrics
    """
    # Extract decision variables
    dvars = extract_decision_variables(trace, topo, solver_result.x_outer)

    # Compute metrics
    metrics = compute_metrics(trace, dvars, solver_result.primal_obj)

    return FOOResult(
        dvars=dvars,
        n_requests=metrics['n_requests'],
        n_unique_objects=metrics['n_unique_objects'],
        float_hits=metrics['float_hits'],
        integer_hits=metrics['integer_hits'],
        ohr=metrics['ohr'],
        primal_obj=solver_result.primal_obj,
        iterations=solver_result.iterations,
        converged=solver_result.converged,
        x_outer=np.asarray(solver_result.x_outer)
    )


def write_output(
    path: str,
    trace: TraceData,
    dvars: np.ndarray,
    include_header: bool = True
) -> None:
    """
    Write decision variables to file in format matching C++ FOO.

    Format: timestamp id size dvar

    Args:
        path: Output file path
        trace: Original trace data
        dvars: Decision variables
        include_header: Whether to include header line
    """
    with open(path, 'w') as f:
        if include_header:
            f.write("# timestamp id size dvar\n")

        for i in range(trace.n_requests):
            ts = trace.timestamps[i]
            oid = trace.obj_ids[i]
            size = trace.obj_sizes[i]
            dvar = dvars[i]
            f.write(f"{ts} {oid} {size} {dvar:.6f}\n")


def print_summary(
    result: FOOResult,
    cache_size: int,
    elapsed_time: Optional[float] = None
) -> None:
    """
    Print summary matching C++ FOO output format.

    Args:
        result: FOO result
        cache_size: Cache size used
        elapsed_time: Optional solve time in seconds
    """
    print(f"FOO Result Summary")
    print(f"==================")
    print(f"Cache size: {cache_size:,} bytes")
    print(f"Requests: {result.n_requests:,}")
    print(f"Unique objects: {result.n_unique_objects:,}")
    print(f"")
    print(f"Solver:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations:,}")
    print(f"  Primal objective: {result.primal_obj:.4f}")
    if elapsed_time:
        print(f"  Time: {elapsed_time:.2f}s")
    print(f"")
    print(f"Hit Ratio:")
    print(f"  Float hits: {result.float_hits:.2f}")
    print(f"  Integer hits: {result.integer_hits:,}")
    print(f"  OHR: {result.ohr:.6f} ({result.ohr*100:.2f}%)")
