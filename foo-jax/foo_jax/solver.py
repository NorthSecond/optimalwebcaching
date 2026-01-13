"""
r2HPDHG (Restarted Halpern Primal-Dual Hybrid Gradient) solver for FOO.

This implements the restarted Halpern PDHG algorithm for solving the LP
relaxation of the minimum cost flow problem.

LP formulation:
    minimize    c^T x
    subject to  Ax = b    (flow conservation)
                0 <= x <= u  (capacity bounds)

Reference:
    Lu & Yang, "Restarted Halpern PDHG for Linear Programming", 2024
    cuPDLPx: GPU implementation of rHPDHG
"""

from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
import jax.lax as lax

from .topology import Topology
from .operators import apply_A, apply_A_T


@dataclass
class SolverConfig:
    """Configuration for r2HPDHG solver."""
    max_iters: int = 10000
    tol: float = 1e-4
    step_size: float = 0.1  # σ = τ = step_size (smaller for stability)
    restart_interval: int = 500  # Restart every N iterations
    verbose: bool = False
    check_interval: int = 100  # Check convergence every N iterations


class SolverState(NamedTuple):
    """State for r2HPDHG iteration."""
    # Primal variables
    x_inner: jnp.ndarray          # float32[n_inner]
    x_outer: jnp.ndarray          # float32[n_outer]

    # Dual variables
    y: jnp.ndarray                # float32[n_nodes]

    # Extrapolated primal
    x_inner_bar: jnp.ndarray
    x_outer_bar: jnp.ndarray

    # Initial point (for Halpern)
    x_inner_0: jnp.ndarray
    x_outer_0: jnp.ndarray

    # Iteration counter
    k: int


class SolverResult(NamedTuple):
    """Result from r2HPDHG solver."""
    x_inner: jnp.ndarray
    x_outer: jnp.ndarray
    y: jnp.ndarray
    primal_obj: float
    dual_obj: float
    primal_residual: float
    dual_residual: float
    iterations: int
    converged: bool


def initialize_state(topo: Topology) -> SolverState:
    """
    Initialize solver state for PDHG.

    Strategy: Start with x_outer = 0 (optimal caching intention).
    This is typically infeasible when cache overflows, but PDHG will
    naturally adjust towards feasibility while minimizing cost.

    The key insight is that the cost term (1/size for each outer arc)
    naturally pushes x_outer towards 0 (caching), while flow conservation
    constraints push it up where needed (eviction).

    Starting from x_outer = 0 ensures the Halpern anchor is at the
    cost-optimal point, so the algorithm stays near optimal.
    """
    n_inner = topo.n_inner
    n_outer = topo.n_outer
    n_nodes = topo.n_nodes

    # Start with x_outer = 0 (try to cache everything)
    x_outer = jnp.zeros(n_outer, dtype=jnp.float32)

    # Compute x_inner for flow conservation (ignoring capacity for init)
    supply = jnp.array(topo.supply, dtype=jnp.float32)
    cumsum = jnp.cumsum(supply[:-1])
    x_inner = cumsum.astype(jnp.float32)

    # Clip x_inner to capacity bounds (may create infeasibility)
    # PDHG will resolve this through dual updates
    x_inner = jnp.clip(x_inner, 0.0, topo.inner_capacity)

    y = jnp.zeros(n_nodes, dtype=jnp.float32)

    return SolverState(
        x_inner=x_inner,
        x_outer=x_outer,
        y=y,
        x_inner_bar=x_inner,
        x_outer_bar=x_outer,
        x_inner_0=x_inner,
        x_outer_0=x_outer,
        k=0
    )


def _pdhg_step(
    state: SolverState,
    sigma: float,
    tau: float,
    supply: jnp.ndarray,
    outer_from: jnp.ndarray,
    outer_to: jnp.ndarray,
    outer_cost: jnp.ndarray,
    inner_capacity: float,
    outer_capacity: jnp.ndarray,
    # Per-arc step size scaling for outer arcs
    outer_step_scale: jnp.ndarray = None
) -> SolverState:
    """
    Single r2HPDHG iteration.

    Algorithm:
        1. Dual update: y_{k+1} = y_k + σ * (A @ x_bar_k - b)
        2. Primal update: x_{k+1} = clip(x_k - τ * (A^T @ y_{k+1} + c), 0, cap)
        3. Halpern extrapolation: x_bar_{k+1} = α * (2*x_{k+1} - x_k) + (1-α) * x_0

    Note: outer_step_scale can be used to scale the step size per arc.
    Set to capacity to normalize the cost term (cost * capacity = 1).
    """
    k = state.k

    # 1. Dual update: y_{k+1} = y_k + σ * (A @ x_bar_k - b)
    residual = apply_A(
        state.x_inner_bar, state.x_outer_bar,
        outer_from, outer_to, len(supply)
    )
    y_new = state.y + sigma * (residual - supply)

    # 2. Primal update: x_{k+1} = clip(x_k - τ * (A^T @ y_{k+1} + c), 0, cap)
    grad_inner, grad_outer = apply_A_T(y_new, outer_from, outer_to)

    # Inner arcs: cost = 0
    x_inner_pre = state.x_inner - tau * grad_inner
    x_inner_new = jnp.clip(x_inner_pre, 0.0, inner_capacity)

    # Outer arcs: use per-arc scaling if provided
    # This helps with numerical stability when costs are very small
    if outer_step_scale is not None:
        # Scale the cost term by capacity to get uniform updates
        # grad term gets regular tau, cost term gets tau * scale
        x_outer_pre = state.x_outer - tau * grad_outer - tau * outer_step_scale * outer_cost
    else:
        x_outer_pre = state.x_outer - tau * (grad_outer + outer_cost)
    x_outer_new = jnp.clip(x_outer_pre, 0.0, outer_capacity)

    # 3. Halpern extrapolation
    alpha = (k + 1.0) / (k + 2.0)

    x_inner_bar_new = alpha * (2 * x_inner_new - state.x_inner) + (1 - alpha) * state.x_inner_0
    x_outer_bar_new = alpha * (2 * x_outer_new - state.x_outer) + (1 - alpha) * state.x_outer_0

    return SolverState(
        x_inner=x_inner_new,
        x_outer=x_outer_new,
        y=y_new,
        x_inner_bar=x_inner_bar_new,
        x_outer_bar=x_outer_bar_new,
        x_inner_0=state.x_inner_0,
        x_outer_0=state.x_outer_0,
        k=k + 1
    )


def compute_residuals(
    state: SolverState,
    supply: jnp.ndarray,
    outer_from: jnp.ndarray,
    outer_to: jnp.ndarray,
    outer_cost: jnp.ndarray,
    inner_capacity: float,
    outer_capacity: jnp.ndarray
) -> Tuple[float, float]:
    """
    Compute primal and dual residuals for convergence checking.

    Primal residual: ||A @ x - b||_2 / (1 + ||b||_2)
    Dual residual: ||clip(x - (A^T @ y + c), 0, cap) - x||_2 / (1 + ||x||_2)
    """
    # Primal residual: flow conservation violation
    Ax = apply_A(
        state.x_inner, state.x_outer,
        outer_from, outer_to, len(supply)
    )
    primal_res = Ax - supply
    primal_norm = jnp.linalg.norm(primal_res)
    primal_rel = primal_norm / (1.0 + jnp.linalg.norm(supply))

    # Dual residual: optimality condition violation
    grad_inner, grad_outer = apply_A_T(state.y, outer_from, outer_to)

    # For inner arcs
    x_inner_proj = jnp.clip(state.x_inner - grad_inner, 0.0, inner_capacity)
    dual_inner = state.x_inner - x_inner_proj

    # For outer arcs
    x_outer_proj = jnp.clip(state.x_outer - (grad_outer + outer_cost), 0.0, outer_capacity)
    dual_outer = state.x_outer - x_outer_proj

    dual_norm = jnp.sqrt(jnp.sum(dual_inner**2) + jnp.sum(dual_outer**2))
    x_norm = jnp.sqrt(jnp.sum(state.x_inner**2) + jnp.sum(state.x_outer**2))
    dual_rel = dual_norm / (1.0 + x_norm)

    return float(primal_rel), float(dual_rel)


def compute_objectives(
    state: SolverState,
    supply: jnp.ndarray,
    outer_cost: jnp.ndarray
) -> Tuple[float, float]:
    """
    Compute primal and dual objective values.

    Primal: c^T x = sum(outer_cost * x_outer)  (inner cost is 0)
    Dual: b^T y = sum(supply * y)
    """
    primal_obj = float(jnp.dot(outer_cost, state.x_outer))
    dual_obj = float(jnp.dot(supply, state.y))
    return primal_obj, dual_obj


def solve(
    topo: Topology,
    config: Optional[SolverConfig] = None,
    progress_callback: Optional[callable] = None
) -> SolverResult:
    """
    Solve FOO using r2HPDHG algorithm.

    Note: This first-order method may not reach the exact optimal solution
    as efficiently as specialized MCF solvers like NetworkSimplex. The LP
    relaxation has integral optimal solutions for MCF, but PDHG may converge
    to a fractional point in the optimal face.

    For best results, use sufficient iterations (50K-100K) and appropriate
    step size (0.1 is usually good). The algorithm uses per-arc scaling
    to handle the widely varying object sizes.

    Args:
        topo: Flow network topology
        config: Solver configuration (uses defaults if None)
        progress_callback: Optional callback for progress reporting

    Returns:
        SolverResult with optimal flows and convergence info
    """
    if config is None:
        config = SolverConfig()

    # Initialize state with x_outer = 0 (try to cache everything)
    state = initialize_state(topo)

    # Extract topology arrays
    supply = topo.supply
    outer_from = topo.outer_from
    outer_to = topo.outer_to
    outer_cost = topo.outer_cost
    inner_capacity = topo.inner_capacity
    outer_capacity = topo.outer_capacity

    # Use per-arc scaling to normalize cost term
    # This helps with convergence but may not reach exact optimal
    outer_step_scale = outer_capacity

    sigma = tau = config.step_size
    converged = False
    final_iter = config.max_iters

    for iteration in range(config.max_iters):
        state = _pdhg_step(
            state, sigma, tau,
            supply, outer_from, outer_to, outer_cost,
            inner_capacity, outer_capacity, outer_step_scale
        )

        if iteration > 0 and iteration % config.check_interval == 0:
            primal_res, dual_res = compute_residuals(
                state, supply, outer_from, outer_to, outer_cost,
                inner_capacity, outer_capacity
            )

            if config.verbose and progress_callback:
                primal_obj, dual_obj = compute_objectives(state, supply, outer_cost)
                progress_callback(
                    f"Iter {iteration}: primal_res={primal_res:.2e}, "
                    f"dual_res={dual_res:.2e}, obj={primal_obj:.4f}"
                )

            if primal_res < config.tol and dual_res < config.tol:
                converged = True
                final_iter = iteration
                break

        if iteration > 0 and iteration % config.restart_interval == 0:
            state = SolverState(
                x_inner=state.x_inner,
                x_outer=state.x_outer,
                y=state.y,
                x_inner_bar=state.x_inner,
                x_outer_bar=state.x_outer,
                x_inner_0=state.x_inner,
                x_outer_0=state.x_outer,
                k=0
            )

    primal_res, dual_res = compute_residuals(
        state, supply, outer_from, outer_to, outer_cost,
        inner_capacity, outer_capacity
    )
    primal_obj, dual_obj = compute_objectives(state, supply, outer_cost)

    return SolverResult(
        x_inner=state.x_inner,
        x_outer=state.x_outer,
        y=state.y,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        primal_residual=primal_res,
        dual_residual=dual_res,
        iterations=final_iter,
        converged=converged
    )


def solve_jit(
    topo: Topology,
    max_iters: int = 10000,
    tol: float = 1e-4,
    step_size: float = 0.1,
    restart_interval: int = 500
) -> SolverResult:
    """
    JIT-compiled solver using lax.scan for maximum efficiency.

    This version compiles the entire iteration loop for better GPU performance.
    Use this for large-scale problems where compilation time is acceptable.

    Key features (matching regular solver):
    - Per-arc scaling: outer_step_scale = outer_capacity (normalizes cost term)
    - Proper initialization: x_outer = 0 (try to cache everything)
    - Halpern extrapolation with periodic restarts

    Args:
        topo: Flow network topology
        max_iters: Maximum iterations
        tol: Convergence tolerance
        step_size: Step size (sigma = tau)
        restart_interval: Restart Halpern anchor every N iterations

    Returns:
        SolverResult with optimal flows and convergence info
    """
    # Extract topology arrays
    supply = topo.supply
    outer_from = topo.outer_from
    outer_to = topo.outer_to
    outer_cost = topo.outer_cost
    inner_capacity = topo.inner_capacity
    outer_capacity = topo.outer_capacity

    # Per-arc scaling to normalize cost term (matches regular solver)
    outer_step_scale = outer_capacity

    sigma = tau = step_size

    # Initialize state using same strategy as initialize_state()
    # x_outer = 0 (try to cache everything)
    x_outer = jnp.zeros(topo.n_outer, dtype=jnp.float32)

    # x_inner from flow conservation (ignoring capacity initially)
    supply_f = jnp.array(supply, dtype=jnp.float32)
    cumsum = jnp.cumsum(supply_f[:-1])
    x_inner = jnp.clip(cumsum, 0.0, inner_capacity)

    y = jnp.zeros(topo.n_nodes, dtype=jnp.float32)

    # JIT-compiled step function with per-arc scaling
    @jax.jit
    def step(carry, _):
        x_inner, x_outer, y, x_inner_bar, x_outer_bar, x_inner_0, x_outer_0, k = carry

        # Dual update: y_{k+1} = y_k + σ * (A @ x_bar_k - b)
        residual = apply_A(x_inner_bar, x_outer_bar, outer_from, outer_to, topo.n_nodes)
        y_new = y + sigma * (residual - supply)

        # Primal update with per-arc scaling for outer arcs
        grad_inner, grad_outer = apply_A_T(y_new, outer_from, outer_to)

        # Inner arcs: cost = 0
        x_inner_new = jnp.clip(x_inner - tau * grad_inner, 0.0, inner_capacity)

        # Outer arcs: use per-arc scaling (cost * capacity normalization)
        x_outer_pre = x_outer - tau * grad_outer - tau * outer_step_scale * outer_cost
        x_outer_new = jnp.clip(x_outer_pre, 0.0, outer_capacity)

        # Halpern extrapolation
        alpha = (k + 1.0) / (k + 2.0)
        x_inner_bar_new = alpha * (2 * x_inner_new - x_inner) + (1 - alpha) * x_inner_0
        x_outer_bar_new = alpha * (2 * x_outer_new - x_outer) + (1 - alpha) * x_outer_0

        new_carry = (x_inner_new, x_outer_new, y_new, x_inner_bar_new, x_outer_bar_new,
                     x_inner_0, x_outer_0, k + 1)
        return new_carry, None

    # JIT-compiled restart function
    @jax.jit
    def do_restart(state):
        x_inner, x_outer, y, _, _, _, _, k = state
        return (x_inner, x_outer, y, x_inner, x_outer, x_inner, x_outer, 0)

    # Run iterations with periodic restarts
    n_blocks = max_iters // restart_interval
    remainder = max_iters % restart_interval

    state = (x_inner, x_outer, y, x_inner, x_outer, x_inner, x_outer, 0)

    for _ in range(n_blocks):
        state, _ = lax.scan(step, state, None, length=restart_interval)
        state = do_restart(state)

    if remainder > 0:
        state, _ = lax.scan(step, state, None, length=remainder)

    x_inner, x_outer, y, _, _, _, _, _ = state

    # Compute final residuals and objectives
    primal_res, dual_res = compute_residuals(
        SolverState(x_inner, x_outer, y, x_inner, x_outer, x_inner, x_outer, 0),
        supply, outer_from, outer_to, outer_cost, inner_capacity, outer_capacity
    )
    primal_obj = float(jnp.dot(outer_cost, x_outer))
    dual_obj = float(jnp.dot(supply, y))

    converged = primal_res < tol and dual_res < tol

    return SolverResult(
        x_inner=x_inner,
        x_outer=x_outer,
        y=y,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        primal_residual=primal_res,
        dual_residual=dual_res,
        iterations=max_iters,
        converged=converged
    )
