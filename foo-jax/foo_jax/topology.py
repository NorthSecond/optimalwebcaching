"""
Flow network topology builder for FOO algorithm.

Converts trace data into the MCF (Min-Cost Flow) network structure matching
the C++ createMCF() function in OHRgoal/FOO/lib/parse_trace.cpp.

The network has:
- Nodes: Created at each request that has a future occurrence
- Inner arcs: Chain of cache capacity constraints between consecutive nodes
- Outer arcs: Object interval arcs connecting repeated accesses
"""

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp

from .trace_parser import TraceData


@dataclass
class Topology:
    """
    Flow network topology for FOO algorithm.

    This structure enables matrix-free computation of A @ x and A.T @ y
    without storing the full constraint matrix.

    Attributes:
        n_nodes: Number of nodes in the network
        supply: Flow conservation RHS (b vector), shape (n_nodes,)

        n_inner: Number of inner arcs (= n_nodes - 1)
        inner_capacity: Cache size (constant for all inner arcs)

        n_outer: Number of outer arcs (one per object interval)
        outer_from: Source node indices, shape (n_outer,)
        outer_to: Destination node indices, shape (n_outer,)
        outer_capacity: Object sizes (arc capacities), shape (n_outer,)
        outer_cost: 1/size (arc costs), shape (n_outer,)
        outer_trace_idx: Trace index for output mapping, shape (n_outer,)

        cache_size: Original cache size parameter
    """
    # Node information
    n_nodes: int
    supply: jnp.ndarray           # float32[n_nodes]

    # Inner arcs (implicit chain: node i -> node i+1)
    n_inner: int                  # = n_nodes - 1
    inner_capacity: float         # = cache_size (constant)

    # Outer arcs (explicit sparse structure)
    n_outer: int
    outer_from: jnp.ndarray       # int32[n_outer]
    outer_to: jnp.ndarray         # int32[n_outer]
    outer_capacity: jnp.ndarray   # float32[n_outer]
    outer_cost: jnp.ndarray       # float32[n_outer]
    outer_trace_idx: jnp.ndarray  # int32[n_outer]

    # Problem parameters
    cache_size: int


class TopologyStats(NamedTuple):
    """Statistics about the built topology."""
    n_nodes: int
    n_inner_arcs: int
    n_outer_arcs: int
    total_supply: float
    total_demand: float


def build_topology(trace: TraceData, cache_size: int) -> Topology:
    """
    Build flow network topology from trace data.

    Matches C++ createMCF() function exactly:
    1. Initial node created
    2. For each request with hasNext=True: create new node + inner arc
    3. For each repeated access: create outer arc from first to second occurrence

    Args:
        trace: Parsed trace data with next_access_idx
        cache_size: Cache capacity in bytes

    Returns:
        Topology structure for matrix-free operators
    """
    n_requests = trace.n_requests
    has_next = trace.next_access_idx >= 0

    # Count nodes: 1 initial + number of requests with has_next=True
    n_nodes = 1 + int(has_next.sum())

    # Count outer arcs: one for each request that has a previous occurrence
    has_prev = trace.prev_access_idx >= 0
    n_outer = int(has_prev.sum())

    # Build node mapping: trace_idx -> node_id
    # Only requests with has_next=True get nodes assigned
    node_ids = np.full(n_requests, -1, dtype=np.int32)
    cur_node_id = 0  # Initial node

    # Map trace indices to node IDs
    # C++: if(nextRequest) { prevNode = curNode; lastSeen[key] = (i, nodeId); curNode = newNode(); }
    for i in range(n_requests):
        if has_next[i]:
            node_ids[i] = cur_node_id
            cur_node_id += 1

    assert cur_node_id == n_nodes - 1, f"Node count mismatch: {cur_node_id} vs {n_nodes - 1}"

    # Build outer arcs and supplies
    outer_from_list = []
    outer_to_list = []
    outer_capacity_list = []
    outer_cost_list = []
    outer_trace_idx_list = []

    # Supply vector: initialized to zero
    supply = np.zeros(n_nodes, dtype=np.float32)

    # Track current node (starts at initial node 0)
    cur_node = 0

    for i in range(n_requests):
        obj_size = int(trace.obj_sizes[i])
        prev_idx = int(trace.prev_access_idx[i])

        # Check if previous interval ended here (object was seen before)
        if prev_idx >= 0:
            # Get node ID where object was first seen
            from_node = node_ids[prev_idx]
            assert from_node >= 0, f"Previous node not found for trace idx {prev_idx}"

            # Create outer arc: from_node -> cur_node
            outer_from_list.append(from_node)
            outer_to_list.append(cur_node)
            outer_capacity_list.append(float(obj_size))
            outer_cost_list.append(1.0 / obj_size)
            outer_trace_idx_list.append(prev_idx)  # Arc belongs to FIRST occurrence

            # Update supplies
            # C++: supplies[lastReq] += size; supplies[curNode] -= size;
            supply[from_node] += obj_size
            supply[cur_node] -= obj_size

        # If there is another request for this object (hasNext)
        if has_next[i]:
            # Create new node for next time point
            # Inner arc: cur_node -> cur_node + 1 (implicit in chain structure)
            cur_node += 1

    assert cur_node == n_nodes - 1, f"Final node mismatch: {cur_node} vs {n_nodes - 1}"
    assert len(outer_from_list) == n_outer, f"Outer arc count mismatch: {len(outer_from_list)} vs {n_outer}"

    # Convert to JAX arrays
    outer_from = jnp.array(outer_from_list, dtype=jnp.int32)
    outer_to = jnp.array(outer_to_list, dtype=jnp.int32)
    outer_capacity = jnp.array(outer_capacity_list, dtype=jnp.float32)
    outer_cost = jnp.array(outer_cost_list, dtype=jnp.float32)
    outer_trace_idx = jnp.array(outer_trace_idx_list, dtype=jnp.int32)
    supply = jnp.array(supply, dtype=jnp.float32)

    return Topology(
        n_nodes=n_nodes,
        supply=supply,
        n_inner=n_nodes - 1,
        inner_capacity=float(cache_size),
        n_outer=n_outer,
        outer_from=outer_from,
        outer_to=outer_to,
        outer_capacity=outer_capacity,
        outer_cost=outer_cost,
        outer_trace_idx=outer_trace_idx,
        cache_size=cache_size
    )


def get_topology_stats(topo: Topology) -> TopologyStats:
    """Get statistics about the topology for debugging."""
    return TopologyStats(
        n_nodes=topo.n_nodes,
        n_inner_arcs=topo.n_inner,
        n_outer_arcs=topo.n_outer,
        total_supply=float(jnp.sum(jnp.maximum(topo.supply, 0))),
        total_demand=float(jnp.sum(jnp.minimum(topo.supply, 0)))
    )


def validate_topology(topo: Topology) -> bool:
    """
    Validate topology for correctness.

    Returns True if valid, raises AssertionError otherwise.
    """
    # Supply must balance (sum to zero)
    supply_sum = float(jnp.sum(topo.supply))
    assert abs(supply_sum) < 1e-6, f"Supply doesn't balance: sum = {supply_sum}"

    # All outer arc indices must be valid
    assert jnp.all(topo.outer_from >= 0), "Invalid outer_from indices"
    assert jnp.all(topo.outer_from < topo.n_nodes), "outer_from out of range"
    assert jnp.all(topo.outer_to >= 0), "Invalid outer_to indices"
    assert jnp.all(topo.outer_to < topo.n_nodes), "outer_to out of range"

    # Outer arcs must go forward in time (from < to)
    # This is a property of the interval graph
    assert jnp.all(topo.outer_from < topo.outer_to), "Outer arcs must go forward"

    # Capacities must be positive
    assert jnp.all(topo.outer_capacity > 0), "Outer capacities must be positive"
    assert topo.inner_capacity > 0, "Inner capacity must be positive"

    return True
