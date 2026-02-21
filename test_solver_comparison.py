#!/usr/bin/env python
"""
Pure-numpy test: Compare NetworkX solver vs C++ FOO.
Avoids JAX imports entirely for fast startup.
"""
import struct
import numpy as np
import networkx as nx
import time
from dataclasses import dataclass

BINARY_TRACE = '/tmp/test_trace.oracleGeneral.bin'
CACHE_SIZE = 1007

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

def read_and_build():
    with open(BINARY_TRACE, 'rb') as f:
        data = f.read()
    n = len(data) // 24
    ts = np.zeros(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t, oid, sz, _ = struct.unpack_from('<IQIq', data, i * 24)
        ts[i], ids[i], sizes[i] = t, oid, sz
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
    supply = np.zeros(n_nodes, dtype=np.float32)
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
    topo = SimpleTopo(
        n_nodes=n_nodes, supply=supply, n_inner=n_nodes-1,
        inner_capacity=float(CACHE_SIZE), n_outer=len(of_l),
        outer_from=np.array(of_l, dtype=np.int32),
        outer_to=np.array(ot_l, dtype=np.int32),
        outer_capacity=np.array(oc_l, dtype=np.float32),
        outer_cost=np.array(occ_l, dtype=np.float32),
        outer_trace_idx=np.array(oti_l, dtype=np.int32),
        cache_size=CACHE_SIZE,
    )
    return topo, n, n_unique, ts, ids, sizes

def solve_multi(topo):
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
                       capacity=float(topo.outer_capacity[j]), weight=float(topo.outer_cost[j]))
        ok.append((int(topo.outer_from[j]), int(topo.outer_to[j]), k))
    fc, fd = nx.network_simplex(G)
    xo = np.zeros(topo.n_outer, dtype=np.float32)
    for j, (u, v, k) in enumerate(ok):
        if u in fd and v in fd[u] and k in fd[u][v]:
            xo[j] = fd[u][v][k]
    return fc, xo

def solve_digraph(topo):
    G = nx.DiGraph()
    for i in range(topo.n_nodes):
        G.add_node(i, demand=-float(topo.supply[i]))
    for i in range(topo.n_inner):
        G.add_edge(i, i+1, capacity=float(topo.inner_capacity), weight=0.0)
    for j in range(topo.n_outer):
        G.add_edge(int(topo.outer_from[j]), int(topo.outer_to[j]),
                   capacity=float(topo.outer_capacity[j]), weight=float(topo.outer_cost[j]))
    try:
        fc, fd = nx.network_simplex(G)
    except Exception as e:
        return None, None, str(e)
    xo = np.zeros(topo.n_outer, dtype=np.float32)
    for j in range(topo.n_outer):
        fn, tn = int(topo.outer_from[j]), int(topo.outer_to[j])
        if fn in fd and tn in fd[fn]:
            xo[j] = fd[fn][tn]
    return fc, xo, None

def dvars_from(topo, xo, n):
    d = np.zeros(n, dtype=np.float32)
    for i in range(topo.n_outer):
        d[topo.outer_trace_idx[i]] = (topo.outer_capacity[i] - xo[i]) / topo.outer_capacity[i]
    return d

def main():
    print("=" * 70)
    print("NetworkX Solver Correctness Test (Pure Numpy)")
    print("=" * 70)
    start = time.time()
    topo, nr, nu, ts, ids, sizes = read_and_build()
    bt = time.time() - start
    np_ = sum(1 for j in range(topo.n_outer) if topo.outer_to[j]-topo.outer_from[j]==1)
    print(f"\n  {nr:,} req, {nu:,} unique, {topo.n_nodes:,} nodes, {topo.n_inner+topo.n_outer:,} arcs")
    print(f"  Parallel arcs (inner+outer overlap): {np_}")
    print(f"  Build: {bt:.3f}s")

    print("\n[MultiDiGraph - FIXED]")
    s = time.time(); fc, xo = solve_multi(topo); t1 = time.time()-s
    d = dvars_from(topo, xo, nr)
    h = nr - nu - fc; ohr = h/nr; ih = int(np.sum(d > 0.99))
    print(f"  OHR={ohr:.6f} hitc={h:.6f} int_hits={ih} cost={fc:.6f} time={t1:.3f}s")

    print("\n[DiGraph - BUGGY]")
    s = time.time(); bfc, bxo, err = solve_digraph(topo); t2 = time.time()-s
    if err:
        print(f"  ERROR: {err}")
    else:
        bd = dvars_from(topo, bxo, nr)
        bh = nr - nu - bfc; bohr = bh/nr; bih = int(np.sum(bd > 0.99))
        print(f"  OHR={bohr:.6f} hitc={bh:.6f} int_hits={bih} cost={bfc:.6f} time={t2:.3f}s")

    print("\n[C++ FOO reference]")
    co, ch, ci, ct = 0.792988, 7887.851107, 7357, 0.080
    print(f"  OHR={co:.6f} hitc={ch:.6f} int_hits={ci} time={ct:.3f}s")

    print(f"\n{'='*70}")
    print(f"  Fixed vs C++: OHR diff = {abs(ohr-co):.8f}")
    if not err:
        print(f"  Buggy vs C++: OHR diff = {abs(bohr-co):.8f}")
        if abs(bohr-co) > abs(ohr-co):
            print(f"  ✓ MultiDiGraph fix improves accuracy")
    if abs(ohr-co) < 1e-4:
        print(f"  ✓ PASS: OHR matches C++ within 1e-4")
    else:
        print(f"  ✗ FAIL: OHR mismatch")
    print(f"  Speed: C++ {ct:.3f}s vs NX {t1:.3f}s ({t1/ct:.0f}x slower)")

if __name__ == "__main__":
    main()
