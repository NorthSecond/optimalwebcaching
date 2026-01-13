"""
Implicit matrix operators for FOO algorithm.

These functions compute A @ x and A.T @ y without storing the constraint matrix,
enabling memory-efficient GPU computation for large-scale problems.

The constraint matrix A represents flow conservation:
    For each node i: sum(inflow) - sum(outflow) = supply[i]

A has structure:
    - Inner arcs: form a chain (node i -> node i+1)
    - Outer arcs: sparse connections based on object access patterns
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from .topology import Topology


@jax.jit
def apply_A(
    x_inner: jnp.ndarray,
    x_outer: jnp.ndarray,
    outer_from: jnp.ndarray,
    outer_to: jnp.ndarray,
    n_nodes: int
) -> jnp.ndarray:
    """
    Compute A @ x = flow balance residuals at each node (matrix-free).

    For node i:
        residual[i] = sum(outflow from i) - sum(inflow to i)

    Inner arc contribution (chain structure):
        - Node i sends x_inner[i] to node i+1 (outflow from i)
        - Node i receives x_inner[i-1] from node i-1 (inflow to i)

    Outer arc contribution (sparse):
        - Node outer_from[j] sends x_outer[j] (outflow)
        - Node outer_to[j] receives x_outer[j] (inflow)

    Args:
        x_inner: Flow on inner arcs, shape (n_inner,) where n_inner = n_nodes - 1
        x_outer: Flow on outer arcs, shape (n_outer,)
        outer_from: Source node indices for outer arcs
        outer_to: Destination node indices for outer arcs
        n_nodes: Total number of nodes (used for output array size)

    Returns:
        residual: Flow balance at each node, shape (n_nodes,)
                  residual = A @ x (positive = excess outflow)
    """
    # Infer n_nodes from x_inner shape to avoid tracing issues
    # n_inner = n_nodes - 1, so n_nodes = n_inner + 1
    n_nodes_inferred = x_inner.shape[0] + 1

    residual = jnp.zeros(n_nodes_inferred, dtype=jnp.float32)

    # Inner arc contribution (chain)
    # Outflow: node i sends x_inner[i] for i in [0, n_nodes-2]
    residual = residual.at[:-1].add(x_inner)
    # Inflow: node i receives x_inner[i-1] for i in [1, n_nodes-1]
    residual = residual.at[1:].add(-x_inner)

    # Outer arc contribution (scatter)
    # Outflow: outer_from[j] sends x_outer[j]
    residual = residual.at[outer_from].add(x_outer)
    # Inflow: outer_to[j] receives x_outer[j]
    residual = residual.at[outer_to].add(-x_outer)

    return residual


@jax.jit
def apply_A_T(
    y: jnp.ndarray,
    outer_from: jnp.ndarray,
    outer_to: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute A.T @ y = gradients for primal variables (matrix-free).

    For inner arc i (node i -> node i+1):
        grad_inner[i] = y[i] - y[i+1]  (since arc contributes +1 at source, -1 at dest)

    For outer arc j (outer_from[j] -> outer_to[j]):
        grad_outer[j] = y[outer_from[j]] - y[outer_to[j]]

    Args:
        y: Dual variables (node potentials), shape (n_nodes,)
        outer_from: Source node indices for outer arcs
        outer_to: Destination node indices for outer arcs

    Returns:
        (grad_inner, grad_outer):
            grad_inner: Gradients for inner arcs, shape (n_inner,)
            grad_outer: Gradients for outer arcs, shape (n_outer,)
    """
    # Inner gradients: simple diff (chain structure)
    # Arc i goes from node i to node i+1
    # A.T contribution: y[i] - y[i+1]
    grad_inner = y[:-1] - y[1:]

    # Outer gradients: gather (sparse structure)
    # Arc j goes from outer_from[j] to outer_to[j]
    # A.T contribution: y[outer_from[j]] - y[outer_to[j]]
    grad_outer = y[outer_from] - y[outer_to]

    return grad_inner, grad_outer


def build_explicit_A(topo: Topology) -> jnp.ndarray:
    """
    Build explicit constraint matrix A for validation (small problems only!).

    A has shape (n_nodes, n_arcs) where n_arcs = n_inner + n_outer.
    For each arc (u -> v):
        A[u, arc_id] = +1  (outflow)
        A[v, arc_id] = -1  (inflow)

    WARNING: Only use for small problems to validate implicit operators.
    For n_nodes > 10000, this will use excessive memory.
    """
    n_nodes = topo.n_nodes
    n_inner = topo.n_inner
    n_outer = topo.n_outer
    n_arcs = n_inner + n_outer

    A = jnp.zeros((n_nodes, n_arcs), dtype=jnp.float32)

    # Inner arcs (chain): arc i connects node i to node i+1
    for i in range(n_inner):
        A = A.at[i, i].set(1.0)      # outflow from node i
        A = A.at[i+1, i].set(-1.0)   # inflow to node i+1

    # Outer arcs: arc j connects outer_from[j] to outer_to[j]
    for j in range(n_outer):
        arc_id = n_inner + j
        from_node = int(topo.outer_from[j])
        to_node = int(topo.outer_to[j])
        A = A.at[from_node, arc_id].set(1.0)   # outflow
        A = A.at[to_node, arc_id].set(-1.0)    # inflow

    return A


def validate_operators(topo: Topology, rtol: float = 1e-5) -> bool:
    """
    Validate implicit operators against explicit matrix multiplication.

    Tests:
    1. apply_A(x_inner, x_outer) == A @ x
    2. apply_A_T(y) == A.T @ y
    3. Adjoint property: <y, A@x> == <A.T@y, x>

    Returns True if all tests pass.
    """
    # Build explicit matrix
    A = build_explicit_A(topo)
    n_inner = topo.n_inner
    n_outer = topo.n_outer

    # Generate random test vectors
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    x_inner = jax.random.uniform(key1, (n_inner,))
    x_outer = jax.random.uniform(key2, (n_outer,))
    y = jax.random.uniform(key3, (topo.n_nodes,))

    # Test apply_A
    x = jnp.concatenate([x_inner, x_outer])
    expected_Ax = A @ x
    actual_Ax = apply_A(x_inner, x_outer, topo.outer_from, topo.outer_to, topo.n_nodes)

    if not jnp.allclose(expected_Ax, actual_Ax, rtol=rtol):
        print(f"apply_A mismatch:")
        print(f"  Expected: {expected_Ax}")
        print(f"  Actual: {actual_Ax}")
        print(f"  Max diff: {jnp.max(jnp.abs(expected_Ax - actual_Ax))}")
        return False

    # Test apply_A_T
    expected_ATy = A.T @ y
    grad_inner, grad_outer = apply_A_T(y, topo.outer_from, topo.outer_to)
    actual_ATy = jnp.concatenate([grad_inner, grad_outer])

    if not jnp.allclose(expected_ATy, actual_ATy, rtol=rtol):
        print(f"apply_A_T mismatch:")
        print(f"  Expected: {expected_ATy}")
        print(f"  Actual: {actual_ATy}")
        print(f"  Max diff: {jnp.max(jnp.abs(expected_ATy - actual_ATy))}")
        return False

    # Test adjoint property: <y, Ax> == <A.T y, x>
    lhs = jnp.dot(y, actual_Ax)
    rhs = jnp.dot(actual_ATy, x)

    if not jnp.isclose(lhs, rhs, rtol=rtol):
        print(f"Adjoint property failed:")
        print(f"  <y, Ax> = {lhs}")
        print(f"  <A.T y, x> = {rhs}")
        return False

    return True
