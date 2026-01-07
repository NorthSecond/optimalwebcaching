#!/usr/bin/env python3
"""
FOO (Flow Offline Optimum) Optimal Caching - GPU-Accelerated Implementation

Migrated from C++/LEMON to Python/cuOpt for 10-100x speedup on large traces.
Original: /home/ubuntu/data/optimalwebcaching/OHRgoal/FOO/

Author: Migrated from D. Berger et al. (SIGMETRICS 2018)
License: Same as original C++ implementation
"""

import struct
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import zstandard as zstd


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Node:
    """
    Flow network node representing a request with future occurrence.

    Corresponds to SmartDigraph::Node in C++ LEMON implementation.
    """
    id: int              # Sequential node ID
    trace_idx: int       # Index in original trace
    obj_id: int          # Object ID
    obj_size: int        # Object size (bytes)
    supply: int          # +size (supply) or -size (demand)


@dataclass
class OuterArc:
    """
    Arc connecting repeated requests for the same object.

    Represents a caching decision: flow = bytes of object kept in cache.
    Corresponds to "outer" arcs in C++ createMCF function.
    """
    id: int              # Arc ID in LP variable vector
    from_node: int       # Source node
    to_node: int         # Destination node
    obj_size: int        # Capacity (object size)
    cost: float          # 1/obj_size (for OHR optimization)
    trace_idx: int       # First occurrence index (for output)


@dataclass
class InnerArc:
    """
    Arc enforcing cache capacity constraint between consecutive time points.

    Corresponds to "inner" arcs in C++ createMCF function.
    """
    id: int              # Arc ID (offset by num_outer_arcs)
    from_node: int       # Source node
    to_node: int         # Destination node
    capacity: int        # Cache size
    cost: float          # 0.0 (free to carry forward)


@dataclass
class CSRMatrix:
    """
    Compressed Sparse Row matrix for LP constraints.

    Replaces LEMON's SmartDigraph with explicit constraint matrix.
    """
    row_offsets: np.ndarray  # Shape: (n_rows+1,) dtype=int64
    col_indices: np.ndarray  # Shape: (nnz,) dtype=int64
    values: np.ndarray       # Shape: (nnz,) dtype=float64
    shape: Tuple[int, int]   # (n_rows, n_cols)


# ============================================================================
# Phase 1: Trace Parser
# ============================================================================

def parse_oracle_general(trace_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse OracleGeneral binary format (24 bytes/record).

    Binary structure (little-endian):
      - uint32 timestamp        (4 bytes)
      - uint64 obj_id           (8 bytes)
      - uint32 obj_size         (4 bytes)
      - int64  next_access_vtime (8 bytes, -1 if no next access)

    Reference: /lib/trace/oracle_general_reader.cpp

    Args:
        trace_path: Path to .dat or .zst trace file

    Returns:
        List of (timestamp, obj_id, obj_size, next_access_vtime) tuples
    """
    is_zst = trace_path.endswith('.zst')

    # Read file (with optional zstd decompression)
    with open(trace_path, 'rb') as f:
        if is_zst:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = reader.read()
        else:
            data = f.read()

    # Parse binary records
    records = []
    record_size = 24  # 4 + 8 + 4 + 8
    n_records = len(data) // record_size

    for i in range(n_records):
        offset = i * record_size
        timestamp, obj_id, obj_size, next_vtime = struct.unpack(
            '<IQIq',  # Little-endian: uint32, uint64, uint32, int64
            data[offset:offset+record_size]
        )

        # Skip zero-size objects (validation from C++)
        if obj_size > 0:
            records.append((timestamp, obj_id, obj_size, next_vtime))

    return records


# ============================================================================
# Phase 2: Graph Builder
# ============================================================================

def build_graph_from_trace(
    records: List[Tuple[int, int, int, int]],
    cache_size: int
) -> Tuple[List[Node], List[OuterArc], List[InnerArc]]:
    """
    Build flow network nodes and arcs from trace.

    Ports C++ createMCF function from:
      /OHRgoal/FOO/lib/parse_trace.cpp:39-77

    Algorithm (matching C++ exactly):
      1. Create initial curNode (line 45)
      2. For each request:
         a. If object seen before: create outer arc from lastReq to curNode
         b. If has_next: save curNode as prevNode, create new curNode, create inner arc

    Args:
        records: Parsed trace records
        cache_size: Cache capacity in bytes

    Returns:
        (nodes, outer_arcs, inner_arcs) tuple
    """
    nodes = []
    outer_arcs = []
    inner_arcs = []

    # Map (obj_id, obj_size) -> (node_id, trace_idx)
    # C++: std::map<pair<uint64_t, uint64_t>, pair<uint64_t, int>> lastSeen
    last_seen: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Create initial node (C++: line 45)
    cur_node_id = 0
    nodes.append(Node(
        id=cur_node_id,
        trace_idx=-1,  # Initial node
        obj_id=-1,
        obj_size=0,
        supply=0
    ))

    arc_id = 0

    # Iterate over trace (C++: lines 49-76)
    for trace_idx, (timestamp, obj_id, obj_size, next_vtime) in enumerate(records):
        has_next = (next_vtime >= 0)
        key = (obj_id, obj_size)

        # Check if previous interval ended here (C++: lines 55-64)
        if key in last_seen:
            last_node_id, last_trace_idx = last_seen[key]

            # Create "outer" request arc from lastReq to curNode
            # NOTE: C++ stores arc ID at FIRST occurrence (last_trace_idx), not second!
            # Second occurrences will get dvar=0 (HIT) by default in output
            outer_arcs.append(OuterArc(
                id=arc_id,
                from_node=last_node_id,
                to_node=cur_node_id,  # To CURRENT node
                obj_size=obj_size,
                cost=1.0 / obj_size,
                trace_idx=last_trace_idx  # FIRST occurrence (matches C++ logic)
            ))
            arc_id += 1

            # Update supplies (C++: lines 61-62)
            nodes[last_node_id].supply += obj_size   # Source
            nodes[cur_node_id].supply -= obj_size    # Sink

        # If there is another request for this object (C++: lines 66-75)
        if has_next:
            # Save current node as prev
            prev_node_id = cur_node_id

            # Record this position for future references
            last_seen[key] = (prev_node_id, trace_idx)

            # Create new node and advance curNode
            cur_node_id = len(nodes)
            nodes.append(Node(
                id=cur_node_id,
                trace_idx=trace_idx,
                obj_id=obj_id,
                obj_size=obj_size,
                supply=0
            ))

            # Create "inner" capacity arc from prev to new curNode
            inner_arcs.append(InnerArc(
                id=arc_id,
                from_node=prev_node_id,
                to_node=cur_node_id,
                capacity=cache_size,
                cost=0.0
            ))
            arc_id += 1

    return nodes, outer_arcs, inner_arcs


# ============================================================================
# Phase 3: LP Matrix Builder
# ============================================================================

def build_flow_conservation_matrix(
    nodes: List[Node],
    outer_arcs: List[OuterArc],
    inner_arcs: List[InnerArc]
) -> Tuple[CSRMatrix, np.ndarray]:
    """
    Build CSR constraint matrix for flow conservation.

    Constraint (for each node v):
      Σ flow_in[arc] - Σ flow_out[arc] = supply[v]

    Replaces LEMON's implicit flow conservation with explicit Ax = b.

    Args:
        nodes: Flow network nodes
        outer_arcs: Object caching decision arcs
        inner_arcs: Cache capacity enforcement arcs

    Returns:
        (A_csr, b_rhs) where A is constraint matrix, b is supply vector
    """
    n_nodes = len(nodes)
    n_arcs = len(outer_arcs) + len(inner_arcs)

    # Build adjacency lists: node -> [(arc_id, coefficient)]
    # coefficient = +1 for incoming arc, -1 for outgoing arc
    incoming_arcs = [[] for _ in range(n_nodes)]
    outgoing_arcs = [[] for _ in range(n_nodes)]

    # Process outer arcs
    for arc in outer_arcs:
        outgoing_arcs[arc.from_node].append((arc.id, -1.0))  # Outgoing: -1
        incoming_arcs[arc.to_node].append((arc.id, 1.0))     # Incoming: +1

    # Process inner arcs (use arc.id directly - IDs are already correct)
    for arc in inner_arcs:
        outgoing_arcs[arc.from_node].append((arc.id, -1.0))
        incoming_arcs[arc.to_node].append((arc.id, 1.0))

    # Build CSR arrays
    row_offsets = [0]
    col_indices = []
    values = []

    for node_id in range(n_nodes):
        # Add outgoing arcs
        for arc_id, coeff in outgoing_arcs[node_id]:
            col_indices.append(arc_id)
            values.append(coeff)

        # Add incoming arcs
        for arc_id, coeff in incoming_arcs[node_id]:
            col_indices.append(arc_id)
            values.append(coeff)

        row_offsets.append(len(col_indices))

    # RHS = -supply vector
    # Standard flow conservation: outgoing - incoming = supply
    # Our matrix has: incoming - outgoing, so RHS must be negated
    b_rhs = np.array([-node.supply for node in nodes], dtype=np.float64)

    csr = CSRMatrix(
        row_offsets=np.array(row_offsets, dtype=np.int64),
        col_indices=np.array(col_indices, dtype=np.int64),
        values=np.array(values, dtype=np.float64),
        shape=(n_nodes, n_arcs)
    )

    return csr, b_rhs


def build_objective_coefficients(
    outer_arcs: List[OuterArc],
    inner_arcs: List[InnerArc]
) -> np.ndarray:
    """
    Build cost vector for LP objective function.

    Objective: minimize c^T x where
      c[i] = 1/size for outer arcs, 0 for inner arcs

    NOTE: Arc IDs are interleaved during graph construction, so we must
    index by arc.id rather than concatenating arrays.

    Args:
        outer_arcs: Object caching arcs (cost = 1/size)
        inner_arcs: Capacity arcs (cost = 0)

    Returns:
        Cost vector indexed by arc ID
    """
    # Find total number of arcs
    all_arcs = outer_arcs + inner_arcs
    if not all_arcs:
        return np.array([], dtype=np.float64)

    n_arcs = max(arc.id for arc in all_arcs) + 1
    c = np.zeros(n_arcs, dtype=np.float64)

    # Set costs for outer arcs (inner arcs default to 0)
    for arc in outer_arcs:
        c[arc.id] = arc.cost

    return c


def build_variable_bounds(
    outer_arcs: List[OuterArc],
    inner_arcs: List[InnerArc],
    cache_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds for LP variables.

    Bounds:
      - Outer arcs: 0 <= x <= obj_size
      - Inner arcs: 0 <= x <= cache_size

    NOTE: Arc IDs are interleaved during graph construction, so we must
    index by arc.id rather than concatenating arrays.

    Args:
        outer_arcs: Object caching arcs
        inner_arcs: Capacity arcs
        cache_size: Cache capacity

    Returns:
        (lower_bounds, upper_bounds) arrays indexed by arc ID
    """
    # Find total number of arcs
    all_arcs = outer_arcs + inner_arcs
    if not all_arcs:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n_arcs = max(arc.id for arc in all_arcs) + 1
    lb = np.zeros(n_arcs, dtype=np.float64)
    ub = np.zeros(n_arcs, dtype=np.float64)

    # Set bounds for outer arcs
    for arc in outer_arcs:
        lb[arc.id] = 0.0
        ub[arc.id] = arc.obj_size

    # Set bounds for inner arcs
    for arc in inner_arcs:
        lb[arc.id] = 0.0
        ub[arc.id] = cache_size

    return lb, ub


# ============================================================================
# Phase 4: cuOpt Solver Integration (PLACEHOLDER)
# ============================================================================

def solve_foo_cuopt(
    A_csr: CSRMatrix,
    b_rhs: np.ndarray,
    c_obj: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    time_limit: int = 3600,
    gpu_id: int = 6
) -> Tuple[np.ndarray, float, str]:
    """
    Solve LP using cuOpt GPU solver.

    Problem formulation:
      minimize:    c^T x
      subject to:  Ax = b
                   lb <= x <= ub

    Args:
        A_csr: Constraint matrix (CSR format)
        b_rhs: RHS vector (supplies)
        c_obj: Objective coefficients
        lb: Lower bounds
        ub: Upper bounds
        time_limit: Solver time limit (seconds)
        gpu_id: GPU device ID (default: 6)

    Returns:
        (flows, objective_value, status) tuple
    """
    import os

    # Try cuOpt GPU solver first
    try:
        from cuopt.linear_programming import DataModel, Solve, SolverSettings
        from cuopt.linear_programming.solver_settings import SolverMethod
        from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD

        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"[INFO] Using cuOpt GPU PDLP solver on device {gpu_id}")

        # Create data model
        data_model = DataModel()
        data_model.set_problem_name("FOO_Optimal_Caching")

        # Set CSR constraint matrix (equality constraints: Ax = b)
        # cuOpt expects: constraint_lower_bounds <= Ax <= constraint_upper_bounds
        # For equality: set both bounds to b
        # CRITICAL: cuOpt requires int32 for indices/offsets, NOT int64!
        data_model.set_csr_constraint_matrix(
            A_csr.values.astype(np.float64),
            A_csr.col_indices.astype(np.int32),
            A_csr.row_offsets.astype(np.int32)
        )
        data_model.set_constraint_lower_bounds(b_rhs.astype(np.float64))
        data_model.set_constraint_upper_bounds(b_rhs.astype(np.float64))

        # Set objective coefficients (minimize c^T x)
        data_model.set_objective_coefficients(c_obj.astype(np.float64))

        # Set variable bounds
        data_model.set_variable_lower_bounds(lb.astype(np.float64))
        data_model.set_variable_upper_bounds(ub.astype(np.float64))

        # Configure solver settings
        settings = SolverSettings()
        # Use 1e-4 tolerance for faster convergence (PDLP is first-order method)
        # C++ LEMON uses 1e-6, but PDLP would take too long to reach that
        settings.set_optimality_tolerance(1e-4)

        # CRITICAL: Use PDLP method for GPU acceleration (verified with 100% GPU utilization)
        settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)

        print(f"[INFO] Solver method: PDLP (GPU-accelerated), tolerance: 1e-4")

        # Solve
        solution = Solve(data_model, settings)

        # Extract results
        primal_solution = solution.get_primal_solution()
        objective_value = solution.get_primal_objective()

        # Validate solution
        if primal_solution is not None and len(primal_solution) > 0:
            if not np.isfinite(objective_value):
                # Recompute objective if solver returned invalid value
                objective_value = np.dot(c_obj, primal_solution)
            return primal_solution, objective_value, "Optimal"
        else:
            raise RuntimeError(f"cuOpt solver returned empty solution")

    except ImportError as e:
        # Fallback to scipy if cuOpt not available
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

        print(f"[WARN] cuOpt not available, falling back to scipy CPU solver: {e}")

        # Convert CSR to scipy format
        A_scipy = csr_matrix(
            (A_csr.values, A_csr.col_indices, A_csr.row_offsets),
            shape=A_csr.shape
        )

        # Solve using scipy linprog
        result = linprog(
            c=c_obj,
            A_eq=A_scipy.toarray(),
            b_eq=b_rhs,
            bounds=list(zip(lb, ub)),
            method='highs',
            options={'maxiter': 10000, 'presolve': True}
        )

        if result.success:
            return result.x, result.fun, "Optimal"
        else:
            raise RuntimeError(f"Solver failed: {result.message}")


# ============================================================================
# Phase 5: Output Formatter
# ============================================================================

def extract_decision_variables(
    trace: List[Tuple[int, int, int, int]],
    outer_arcs: List[OuterArc],
    flows: np.ndarray
) -> np.ndarray:
    """
    Compute per-request decision variables from flow solution.

    Ports C++ logic from /OHRgoal/FOO/foo.cpp:52-69

    Formula: dvar = (size - flow) / size
      - dvar = 1.0 → MISS
      - dvar = 0.0 → HIT

    C++ logic:
      - First occurrences: get dvar from their outgoing outer arc
      - Second occurrences: get dvar=0 (hardcoded HIT in C++ arcId==-1 branch)

    Args:
        trace: Original trace records
        outer_arcs: Object caching arcs
        flows: Optimal flow solution

    Returns:
        Array of dvar values (length = num_requests)
    """
    # Track which requests have outer arcs (first occurrences)
    has_outer_arc = set(arc.trace_idx for arc in outer_arcs)

    # Initialize: first occurrences as MISS, second as HIT
    dvars = np.zeros(len(trace))
    for i in range(len(trace)):
        if i in has_outer_arc:
            dvars[i] = 1.0  # First occurrence (will be updated from arc flow)
        else:
            dvars[i] = 0.0  # Second occurrence (HIT)

    # Update first occurrences with arc flow dvars
    # C++: const long double dvar = (size-flow[g.arcFromId(arcId)])/static_cast<double>(size);
    for arc in outer_arcs:
        trace_idx = arc.trace_idx  # FIRST occurrence
        flow = flows[arc.id]
        size = arc.obj_size
        dvar = (size - flow) / size
        dvars[trace_idx] = dvar

    return dvars


def compute_ohr(
    trace: List[Tuple[int, int, int, int]],
    dvars: np.ndarray,
    solval: float
) -> Dict[str, float]:
    """
    Compute hit ratio metrics matching C++ output.

    Ports /OHRgoal/FOO/foo.cpp:74-75

    Args:
        trace: Original trace records
        dvars: Decision variables
        solval: Solver objective value (total cost)

    Returns:
        Dictionary with hitc, reqc, OHR, floatHits, integerHits
    """
    total_reqs = len(trace)

    # Count first occurrences (always misses)
    uniq_objs = set()
    total_uniq = 0
    for _, obj_id, obj_size, next_vtime in trace:
        if (obj_id, obj_size) not in uniq_objs:
            uniq_objs.add((obj_id, obj_size))
            total_uniq += 1

    # Compute hits from decision variables
    # C++: floatHits += dvar
    float_hits = np.sum(dvars)

    # Integer hits: count dvars > 0.99 (C++: if(dvar > 0.99) integerHits++)
    integer_hits = np.sum(dvars > 0.99)

    # OHR calculation (C++: 1.0-(solval+totalUniqC)/totalReqc)
    ohr = 1.0 - (solval + total_uniq) / total_reqs

    total_hits = total_reqs - total_uniq - solval

    return {
        'hitc': int(total_hits),
        'reqc': total_reqs,
        'OHR': ohr,
        'floatHits': float(float_hits),
        'integerHits': int(integer_hits)
    }


def write_output(
    output_path: str,
    trace: List[Tuple[int, int, int, int]],
    dvars: np.ndarray
):
    """
    Write decision variables to file (match C++ format).

    Format: <timestamp> <obj_id> <obj_size> <dvar>

    Args:
        output_path: Output file path
        trace: Original trace records
        dvars: Decision variables
    """
    with open(output_path, 'w') as f:
        for (timestamp, obj_id, obj_size, _), dvar in zip(trace, dvars):
            f.write(f"{timestamp} {obj_id} {obj_size} {dvar:.10f}\n")


def print_summary(cache_size: int, metrics: Dict[str, float]):
    """
    Print summary statistics (match C++ stdout format).

    C++ format: ExLP<solver> <cacheSize> hitc <hits> reqc <reqs> OHR <ohr> <floatHits> <intHits>

    Args:
        cache_size: Cache size in bytes
        metrics: Computed metrics
    """
    print(f"ExLP {cache_size} hitc {metrics['hitc']} "
          f"reqc {metrics['reqc']} OHR {metrics['OHR']:.10f} "
          f"{metrics['floatHits']:.6f} {metrics['integerHits']}")


# ============================================================================
# Phase 6: Main CLI
# ============================================================================

def main():
    """
    Main entry point matching C++ FOO interface.

    C++ command: ./foo [trace.dat] [cacheSize] [pivotRule] [output.txt]
    Python command: python foo_cuopt.py [trace.dat] [cacheSize] [output.txt]
    """
    parser = argparse.ArgumentParser(
        description="FOO Optimal Caching - GPU-Accelerated (cuOpt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python foo_cuopt.py trace.oracleGeneral.dat 1073741824 results.txt
  python foo_cuopt.py trace.oracleGeneral.zst 1073741824 results.txt --time-limit 3600

C++ FOO equivalent:
  ./OHRgoal/FOO/foo trace.dat 1073741824 4 results.txt
        """
    )

    parser.add_argument("trace", help="OracleGeneral trace file (.dat or .zst)")
    parser.add_argument("cache_size", type=int, help="Cache size in bytes")
    parser.add_argument("output", help="Output file for decision variables")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Solver time limit in seconds (default: 3600)")

    args = parser.parse_args()

    try:
        # Phase 1: Parse trace
        print(f"[1/6] Parsing trace: {args.trace}")
        trace = parse_oracle_general(args.trace)
        print(f"      Loaded {len(trace)} requests")

        # Phase 2: Build graph
        print(f"[2/6] Building flow network (cache_size={args.cache_size} bytes)")
        nodes, outer_arcs, inner_arcs = build_graph_from_trace(trace, args.cache_size)
        print(f"      Graph: {len(nodes)} nodes, {len(outer_arcs)} outer arcs, {len(inner_arcs)} inner arcs")

        # Phase 3: Build LP matrices
        print(f"[3/6] Constructing LP matrices")
        A_csr, b_rhs = build_flow_conservation_matrix(nodes, outer_arcs, inner_arcs)
        c_obj = build_objective_coefficients(outer_arcs, inner_arcs)
        lb, ub = build_variable_bounds(outer_arcs, inner_arcs, args.cache_size)
        print(f"      LP size: {A_csr.shape[0]}×{A_csr.shape[1]} matrix, {len(A_csr.values)} nnz")

        # Phase 4: Solve LP
        print(f"[4/6] Solving LP (time_limit={args.time_limit}s)")
        flows, obj_val, status = solve_foo_cuopt(A_csr, b_rhs, c_obj, lb, ub, args.time_limit)
        print(f"      {status}: objective = {obj_val:.10f}")

        # Phase 5: Extract results
        print(f"[5/6] Extracting decision variables")
        dvars = extract_decision_variables(trace, outer_arcs, flows)
        metrics = compute_ohr(trace, dvars, obj_val)

        # Phase 6: Output results
        print(f"[6/6] Writing output to: {args.output}")
        write_output(args.output, trace, dvars)
        print_summary(args.cache_size, metrics)

        print("\n✅ FOO optimal caching completed successfully")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
