"""
NetworkX-based solver for FOO min-cost flow problem.

Uses NetworkX's network_simplex algorithm for exact solution,
matching the C++ FOO implementation using LEMON's NetworkSimplex.

This provides:
1. Exact optimal solution (not approximation like PDHG)
2. Faster convergence for moderate problem sizes
3. Pure Python implementation (no JAX/GPU required)

IMPORTANT: Must use MultiDiGraph because inner arcs (i -> i+1) and outer arcs
can share the same (from, to) pair when an object is accessed at consecutive
topology nodes.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import math
import numpy as np
import networkx as nx
import time

from .topology import Topology


@dataclass
class SolverResult:
    """Result from NetworkX solver."""
    x_inner: np.ndarray       # Inner arc flows (n_nodes - 1,)
    x_outer: np.ndarray       # Outer arc flows (n_outer,)
    flow_cost: float          # Total flow cost
    iterations: int           # Always 1 for exact solver
    converged: bool           # Always True for exact solver
    solve_time: float         # Solve time in seconds

    @property
    def primal_obj(self) -> float:
        """Alias for compatibility with output.process_result()."""
        return self.flow_cost


def solve_networkx(
    topo: Topology,
    progress_callback: Optional[Callable[[str], None]] = None
) -> SolverResult:
    """
    Solve FOO min-cost flow problem using NetworkX network_simplex.

    The network structure:
    - Nodes: n_nodes with supply/demand values
    - Inner arcs: Chain from node i to node i+1, capacity = cache_size, cost = 0
    - Outer arcs: Object interval arcs, capacity = obj_size, cost = 1/obj_size

    Uses MultiDiGraph to correctly handle parallel arcs (inner and outer arcs
    can share the same (from, to) node pair).

    Args:
        topo: Topology structure from build_topology()
        progress_callback: Optional callback for progress messages

    Returns:
        SolverResult with optimal flow values
    """
    start_time = time.time()

    if progress_callback:
        progress_callback(f"Building NetworkX graph ({topo.n_nodes:,} nodes, {topo.n_inner + topo.n_outer:,} arcs)...")

    # MultiDiGraph supports parallel edges between the same node pair.
    # This is required because an outer arc (from, to) can coincide with
    # an inner arc (i, i+1) when an object is accessed at consecutive nodes.
    G = nx.MultiDiGraph()

    # CRITICAL: NetworkX's network_simplex hangs on floating-point inputs.
    # Scale all values to integers. Supply and capacities are already
    # integer-valued (object sizes). Costs (1/size) are scaled by the LCM
    # of all unique object sizes to produce exact integer costs.
    supply = np.array(topo.supply)
    outer_from = np.array(topo.outer_from)
    outer_to = np.array(topo.outer_to)
    outer_capacity = np.array(topo.outer_capacity)

    # Compute LCM of all unique object sizes for cost scaling
    unique_sizes = set(int(c) for c in outer_capacity)
    cost_scale = 1
    for s in unique_sizes:
        cost_scale = cost_scale * s // math.gcd(cost_scale, s)

    # Add nodes with integer demand (= -supply)
    for i in range(topo.n_nodes):
        G.add_node(i, demand=int(-supply[i]))

    # Add inner arcs (chain: i -> i+1)
    # Cost = 0, capacity = cache_size (integer)
    inner_keys = []
    for i in range(topo.n_inner):
        key = G.add_edge(i, i + 1, capacity=int(topo.inner_capacity), weight=0)
        inner_keys.append((i, i + 1, key))

    # Add outer arcs with integer-scaled costs
    # Original cost = 1/size, scaled cost = cost_scale/size (integer)
    outer_keys = []
    for j in range(topo.n_outer):
        sz = int(outer_capacity[j])
        key = G.add_edge(
            int(outer_from[j]),
            int(outer_to[j]),
            capacity=sz,
            weight=int(cost_scale // sz),
        )
        outer_keys.append((int(outer_from[j]), int(outer_to[j]), key))

    build_time = time.time() - start_time
    if progress_callback:
        progress_callback(f"Graph built in {build_time:.2f}s, solving with network_simplex...")

    # Solve using network simplex
    solve_start = time.time()
    try:
        flow_cost, flow_dict = nx.network_simplex(G)
    except nx.NetworkXUnfeasible as e:
        raise ValueError(f"Infeasible flow problem: {e}")
    except nx.NetworkXUnbounded as e:
        raise ValueError(f"Unbounded flow problem: {e}")

    simplex_time = time.time() - solve_start
    if progress_callback:
        progress_callback(f"Network simplex solved in {simplex_time:.2f}s")

    # Extract flows using stored edge keys
    # For MultiDiGraph, flow_dict[u][v] is a dict of {key: flow}
    x_inner = np.zeros(topo.n_inner, dtype=np.float32)
    for i, (u, v, key) in enumerate(inner_keys):
        if u in flow_dict and v in flow_dict[u] and key in flow_dict[u][v]:
            x_inner[i] = flow_dict[u][v][key]

    x_outer = np.zeros(topo.n_outer, dtype=np.float32)
    for j, (u, v, key) in enumerate(outer_keys):
        if u in flow_dict and v in flow_dict[u] and key in flow_dict[u][v]:
            x_outer[j] = flow_dict[u][v][key]

    # Unscale flow cost: real_cost = scaled_cost / cost_scale
    real_flow_cost = flow_cost / cost_scale

    total_time = time.time() - start_time
    if progress_callback:
        progress_callback(f"Total solve time: {total_time:.2f}s")

    return SolverResult(
        x_inner=x_inner,
        x_outer=x_outer,
        flow_cost=real_flow_cost,
        iterations=1,
        converged=True,
        solve_time=total_time
    )


def solve(
    topo: Topology,
    config=None,  # Ignored, for API compatibility
    progress_callback: Optional[Callable[[str], None]] = None
) -> SolverResult:
    """
    Wrapper for API compatibility with PDHG solver.

    Args:
        topo: Topology structure
        config: Ignored (for compatibility)
        progress_callback: Optional callback

    Returns:
        SolverResult
    """
    return solve_networkx(topo, progress_callback)
