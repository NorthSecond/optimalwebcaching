"""
FOO-JAX: GPU-accelerated Flow Offline Optimal caching algorithm.

This package implements the FOO algorithm using JAX and the r2HPDHG
(restarted Halpern Primal-Dual Hybrid Gradient) solver for efficient
GPU computation of optimal caching decisions.

Key Features:
- Matrix-free implicit operators (no explicit constraint matrix storage)
- Supports 100M+ request traces on 96GB GPU memory
- Compatible with OracleGeneral trace format (libCacheSim)

Usage:
    python -m foo_jax solve trace.zst cache_size output.txt
    python -m foo_jax export-pairs trace.zst cache_size pairs.csv

Reference:
    Berger et al., "Practical Bounds on Optimal Caching with Variable Object Sizes"
    SIGMETRICS 2018
"""

__version__ = "0.1.0"
__all__ = [
    "parse_trace",
    "build_topology",
    "solve_foo",
    "extract_decision_variables",
]
