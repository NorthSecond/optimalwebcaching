#!/usr/bin/env python
"""
Small-scale solver comparison: NetworkX MultiDiGraph vs DiGraph vs C++ FOO.
Uses first 1000 requests for fast turnaround.
"""
import struct
import numpy as np
import networkx as nx
import time
import subprocess
import os
from dataclasses import dataclass

SRC_TRACE = '/tmp/test_trace.oracleGeneral.bin'
SMALL_TRACE = '/tmp/test_trace_small.oracleGeneral.bin'
N_REQ = 1000
FOO_BIN = '/home/ubuntu/ssd/optimalwebcaching/OHRgoal/FOO/foo'

@dataclass
class SimpleTopo:
    n_nodes: int
    supply: np.ndarray
    n_inner: int
    inner_capacity: float
    n_outer: int
    outer_from: np.ndarray
    outer_to: np.ndarray
    outer_capacity: np.ndarray
    outer_cost: np.ndarray
    outer_trace_idx: np.ndarray
    cache_size: int

def create_small_trace():
    """Extract first N_REQ requests into a separate binary file."""
    with open(SRC_TRACE, 'rb') as f:
        data = f.read(N_REQ * 24)
    with open(SMALL_TRACE, 'wb') as f:
        f.write(data)
    return data

def parse_trace(data, n):
    ts = np.zeros(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t, oid, sz, _ = struct.unpack_from('<IQIq', data, i * 24)
        ts[i], ids[i], sizes[i] = t, oid, sz
    return ts, ids, sizes

def compute_cache_size(ids, sizes):
    """0.1% of total unique object footprint."""
    unique = {}
    for i in range(len(ids)):
        unique[(int(ids[i]), int(sizes[i]))] = int(sizes[i])
    footprint = sum(unique.values())
    return max(1, int(footprint * 0.001))

def build_topo(n, ids, sizes, cache_size):
    prev_idx = np.full(n, -1, dtype=np.int64)
    next_idx = np.full(n, -1, dtype=np.int64)
    last_seen = {}
    for i in range(n):
        key = (int(ids[i]), int(sizes[i]))
        if key in last_seen:
            prev_idx[i] = last_seen[key]
            next_idx[last_seen[key]] = i
        last_seen[key] = i
    n_unique = len(last_seen)
    has_next = next_idx >= 0
    n_nodes = 1 + int(has_next.sum())
    node_ids = np.full(n, -1, dtype=np.int32)
    cur_id = 0
    for i in range(n):
        if has_next[i]:
            node_ids[i] = cur_id
            cur_id += 1
    supply = np.zeros(n_nodes, dtype=np.float64)
    of_l, ot_l, oc_l, occ_l, oti_l = [], [], [], [], []
    cur_node = 0
    for i in range(n):
        sz = int(sizes[i])
        pi = int(prev_idx[i])
        if pi >= 0:
            fn = node_ids[pi]
            of_l.append(fn); ot_l.append(cur_node)
            oc_l.append(float(sz)); occ_l.append(1.0 / sz)
            oti_l.append(pi)
            supply[fn] += sz; supply[cur_node] -= sz
        if has_next[i]:
            cur_node += 1
    return SimpleTopo(
        n_nodes=n_nodes, supply=supply, n_inner=n_nodes-1,
        inner_capacity=float(cache_size), n_outer=len(of_l),
        outer_from=np.array(of_l, dtype=np.int32),
        outer_to=np.array(ot_l, dtype=np.int32),
        outer_capacity=np.array(oc_l, dtype=np.float64),
        outer_cost=np.array(occ_l, dtype=np.float64),
        outer_trace_idx=np.array(oti_l, dtype=np.int32),
        cache_size=cache_size,
    ), n_unique

def solve_multi(topo):
    """Correct solver using MultiDiGraph (handles parallel arcs)."""
    G = nx.MultiDiGraph()
    for i in range(topo.n_nodes):
        G.add_node(i, demand=-float(topo.supply[i]))
    ik = []
    for i in range(topo.n_inner):
        k = G.add_edge(i, i+1, capacity=float(topo.inner_capacity), weight=0.0)
        ik.append((i, i+1, k))
    ok = []
    for j in range(topo.n_outer):
        k = G.add_edge(int(topo.outer_from[j]), int(topo.outer_to[j]),
                       capacity=float(topo.outer_capacity[j]),
                       weight=float(topo.outer_cost[j]))
        ok.append((int(topo.outer_from[j]), int(topo.outer_to[j]), k))
    fc, fd = nx.network_simplex(G)
    xo = np.zeros(topo.n_outer, dtype=np.float64)
    for j, (u, v, k) in enumerate(ok):
        if u in fd and v in fd[u] and k in fd[u][v]:
            xo[j] = fd[u][v][k]
    return fc, xo

def solve_digraph(topo):
    """Buggy solver using DiGraph (overwrites parallel arcs)."""
    G = nx.DiGraph()
    for i in range(topo.n_nodes):
        G.add_node(i, demand=-float(topo.supply[i]))
    for i in range(topo.n_inner):
        G.add_edge(i, i+1, capacity=float(topo.inner_capacity), weight=0.0)
    for j in range(topo.n_outer):
        G.add_edge(int(topo.outer_from[j]), int(topo.outer_to[j]),
                   capacity=float(topo.outer_capacity[j]),
                   weight=float(topo.outer_cost[j]))
    try:
        fc, fd = nx.network_simplex(G)
    except Exception as e:
        return None, None, str(e)
    xo = np.zeros(topo.n_outer, dtype=np.float64)
    for j in range(topo.n_outer):
        fn, tn = int(topo.outer_from[j]), int(topo.outer_to[j])
        if fn in fd and tn in fd[fn]:
            xo[j] = fd[fn][tn]
    return fc, xo, None

def dvars_from(topo, xo, n):
    d = np.zeros(n, dtype=np.float64)
    for i in range(topo.n_outer):
        idx = topo.outer_trace_idx[i]
        d[idx] = (topo.outer_capacity[i] - xo[i]) / topo.outer_capacity[i]
    return d

def run_cpp_foo(trace_path, cache_size):
    """Run C++ FOO and parse output."""
    out_file = '/tmp/foo_small_dvars.txt'
    cmd = [FOO_BIN, trace_path, str(cache_size), '4', out_file]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr
    # Parse OHR from output
    ohr = None
    for line in stdout.split('\n'):
        if 'OHR' in line:
            parts = line.split()
            for p in parts:
                try:
                    ohr = float(p)
                except ValueError:
                    pass
    return ohr, elapsed, stdout

def main():
    print("=" * 70)
    print(f"Solver Comparison Test ({N_REQ} requests)")
    print("=" * 70)

    # 1. Create small trace
    data = create_small_trace()
    ts, ids, sizes = parse_trace(data, N_REQ)
    cache_size = compute_cache_size(ids, sizes)
    print(f"\n  Requests: {N_REQ}")
    print(f"  Cache size (0.1% footprint): {cache_size}")

    # 2. Build topology
    t0 = time.time()
    topo, n_unique = build_topo(N_REQ, ids, sizes, cache_size)
    bt = time.time() - t0
    n_parallel = sum(1 for j in range(topo.n_outer)
                     if topo.outer_to[j] - topo.outer_from[j] == 1)
    print(f"  Unique objects: {n_unique}")
    print(f"  Nodes: {topo.n_nodes}, Inner arcs: {topo.n_inner}, Outer arcs: {topo.n_outer}")
    print(f"  Parallel arcs (outer overlaps inner): {n_parallel}")
    print(f"  Topology build: {bt:.3f}s")

    # 3. C++ FOO
    print(f"\n[C++ FOO]")
    cpp_ohr, cpp_time, cpp_out = run_cpp_foo(SMALL_TRACE, cache_size)
    print(f"  Raw output: {cpp_out.strip()[:200]}")
    print(f"  Time: {cpp_time:.3f}s")

    # 4. MultiDiGraph (fixed)
    print(f"\n[MultiDiGraph - FIXED]")
    t0 = time.time()
    fc, xo = solve_multi(topo)
    t_multi = time.time() - t0
    d = dvars_from(topo, xo, N_REQ)
    hitc = N_REQ - n_unique - fc
    ohr = hitc / N_REQ
    int_hits = int(np.sum(d > 0.99))
    print(f"  cost={fc:.6f} hitc={hitc:.6f} OHR={ohr:.6f}")
    print(f"  int_hits={int_hits} time={t_multi:.3f}s")

    # 5. DiGraph (buggy)
    print(f"\n[DiGraph - BUGGY]")
    t0 = time.time()
    bfc, bxo, err = solve_digraph(topo)
    t_di = time.time() - t0
    if err:
        print(f"  ERROR: {err}")
    else:
        bd = dvars_from(topo, bxo, N_REQ)
        bhitc = N_REQ - n_unique - bfc
        bohr = bhitc / N_REQ
        bint_hits = int(np.sum(bd > 0.99))
        print(f"  cost={bfc:.6f} hitc={bhitc:.6f} OHR={bohr:.6f}")
        print(f"  int_hits={bint_hits} time={t_di:.3f}s")

    # 6. Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    if not err:
        diff_multi_di = abs(ohr - bohr)
        print(f"  MultiDiGraph OHR: {ohr:.6f}")
        print(f"  DiGraph OHR:      {bohr:.6f}")
        print(f"  Difference:       {diff_multi_di:.8f}")
        if n_parallel > 0 and diff_multi_di > 1e-6:
            print(f"  ✓ MultiDiGraph fix matters: {n_parallel} parallel arcs cause divergence")
        elif n_parallel == 0:
            print(f"  ℹ No parallel arcs in this trace subset — both solvers equivalent")
    print(f"\n  Speed: C++ {cpp_time:.3f}s vs NX-Multi {t_multi:.3f}s ({t_multi/max(cpp_time,0.001):.0f}x slower)")

if __name__ == "__main__":
    main()
