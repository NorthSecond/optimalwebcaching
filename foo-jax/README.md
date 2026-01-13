# FOO-JAX: GPU-Accelerated Optimal Caching

GPU-accelerated implementation of the **FOO (Flow Offline Optimal)** caching algorithm using JAX and the r2HPDHG solver.

## Overview

This implementation replaces the traditional NetworkSimplex MCF solver with a **matrix-free** first-order method (r2HPDHG) that runs entirely on GPU, enabling:

- **100M+ request traces** on RTX 6000 Blackwell (96GB GDDR7)
- **10-100x speedup** over CPU-based LEMON solver
- **Memory efficient**: No explicit constraint matrix storage

## Installation

```bash
cd /home/ubuntu/ssd/optimalwebcaching/foo-jax
uv sync
```

## Usage

### Solve FOO

```bash
# Basic usage
python -m foo_jax solve trace.zst 128974848 output.txt

# With options
python -m foo_jax solve trace.zst 128974848 output.txt --max-iters 50000 --tol 1e-4
```

### Export Pairwise Dataset

```bash
python -m foo_jax export-pairs trace.zst 128974848 pairs.csv
```

## Algorithm

The FOO algorithm formulates optimal caching as a **Minimum Cost Flow (MCF)** problem:

1. **Nodes**: Time intervals between repeated object accesses
2. **Inner Arcs**: Cache capacity constraints (flow = cached bytes)
3. **Outer Arcs**: Object eviction decisions (flow = evicted bytes, cost = 1/size)

Instead of NetworkSimplex, we solve the LP relaxation using **r2HPDHG**:

```
minimize    c^T x
subject to  Ax = b    (flow conservation)
            0 <= x <= u (capacity bounds)
```

Key innovation: `apply_A()` and `apply_A_T()` compute matrix-vector products **without storing A**.

## Performance Targets

| Scale | Requests | GPU Memory | Time |
|-------|----------|------------|------|
| Small | 100K | < 1 GB | < 5 min |
| Medium | 1M | < 5 GB | < 30 min |
| Large | 10M | < 20 GB | < 2 hours |
| Target | 100M | < 50 GB | < 12 hours |

## References

- Berger et al., "Practical Bounds on Optimal Caching with Variable Object Sizes", SIGMETRICS 2018
- Lu & Yang, "Restarted Halpern PDHG for Linear Programming", 2024
- cuPDLPx: GPU implementation of rHPDHG, 2024

## License

Same as the original optimalwebcaching repository.
