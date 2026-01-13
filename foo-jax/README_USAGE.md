# FOO-JAX 使用说明

GPU 加速的 FOO (Flow Offline Optimal) 缓存策略计算工具。

## 快速开始

### 1. 环境激活

```bash
source /home/ubuntu/ssd/optimalwebcaching/foo-cuopt/bin/activate
cd /home/ubuntu/ssd/optimalwebcaching/foo-jax
```

### 2. 基本使用

```python
import sys
sys.path.insert(0, '/home/ubuntu/ssd/optimalwebcaching/foo-jax')

from foo_jax.trace_parser import parse_trace
from foo_jax.topology import build_topology
from foo_jax.solver import solve_jit
from foo_jax.output import process_result

# 解析 trace (推荐使用预解压的 .dat 文件，速度快 2x)
trace = parse_trace(
    '/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.dat',
    max_requests=1_000_000  # 可选：限制请求数
)

# 构建流网络
cache_size = 128_974_848  # 123 MiB
topo = build_topology(trace, cache_size)

# 求解 (GPU 加速)
result = solve_jit(
    topo,
    max_iters=30000,    # 最大迭代次数
    tol=1e-4,           # 收敛精度
    step_size=0.1       # 步长
)

# 计算命中率
foo_result = process_result(trace, topo, result)
print(f"OHR: {foo_result.ohr*100:.2f}%")
print(f"Hits: {foo_result.integer_hits:,}")
```

## Trace 文件

### 可用 trace

| 文件 | 大小 | 请求数 | 推荐 |
|------|------|--------|------|
| `cluster54.oracleGeneral.sample10.dat` | 27 GB | 1.18B | ✅ 预解压，最快 |
| `cluster54.oracleGeneral.sample10.zst` | 8.8 GB | 1.18B | 压缩版 |
| `test_trace_10k.dat` | 240 KB | 10K | 测试用 |

### 路径

```python
# 预解压 trace (推荐)
DAT_PATH = '/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.dat'

# 压缩 trace
ZST_PATH = '/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.zst'

# 测试 trace
TEST_PATH = '/home/ubuntu/ssd/optimalwebcaching/cuopt-python/test_trace_10k.dat'
```

## API 参考

### `parse_trace(path, max_requests=None, use_rust=True)`

解析 OracleGeneral 格式 trace 文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | str | trace 文件路径 (.dat 或 .zst) |
| `max_requests` | int | 最大请求数 (None=全部) |
| `use_rust` | bool | 使用 Rust 加速 (默认 True) |

### `build_topology(trace, cache_size)`

构建最小费用流网络。

| 参数 | 类型 | 说明 |
|------|------|------|
| `trace` | TraceData | 解析后的 trace |
| `cache_size` | int | 缓存大小 (字节) |

### `solve_jit(topo, max_iters, tol, step_size, restart_interval)`

JIT 编译的 GPU 求解器。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topo` | Topology | - | 流网络拓扑 |
| `max_iters` | int | 10000 | 最大迭代次数 |
| `tol` | float | 1e-4 | 收敛精度 |
| `step_size` | float | 0.1 | 步长 (σ=τ) |
| `restart_interval` | int | 500 | Halpern 重启间隔 |

### `solve(topo, config)`

常规求解器 (带进度回调)。

```python
from foo_jax.solver import solve, SolverConfig

config = SolverConfig(
    max_iters=30000,
    tol=1e-4,
    step_size=0.1,
    verbose=True,
    check_interval=500
)
result = solve(topo, config)
```

## 性能参考

### 不同规模性能 (使用 .dat 文件)

| Requests | Parse | Topo | Solve | Total | OHR |
|----------|-------|------|-------|-------|-----|
| 500K | 25s | 2s | 3s | **30s** | 70% |
| 1M | 28s | 4s | 3s | **35s** | 77% |
| 2M | 25s | 8s | 6s | **39s** | 80% |

### GPU 显存占用

| Requests | Nodes | 显存 |
|----------|-------|------|
| 500K | 350K | ~12 MB |
| 1M | 770K | ~26 MB |
| 2M | 1.6M | ~55 MB |
| 5M | 4.3M | ~150 MB |

### 精度

- 与 C++ NetworkSimplex 相比: **~0.5-1% OHR 差距**
- 原因: PDHG 一阶方法可能收敛到分数解

## 完整示例

```python
#!/usr/bin/env python3
"""FOO-JAX 完整示例"""

import sys
import time
sys.path.insert(0, '/home/ubuntu/ssd/optimalwebcaching/foo-jax')

from foo_jax.trace_parser import parse_trace
from foo_jax.topology import build_topology, get_topology_stats
from foo_jax.solver import solve_jit
from foo_jax.output import process_result

def main():
    # 配置
    trace_path = '/home/ubuntu/ssd/libCacheSim/data/twitter/cluster54.oracleGeneral.sample10.dat'
    cache_size = 128_974_848  # 123 MiB
    max_requests = 1_000_000

    print("FOO-JAX Optimal Caching Calculator")
    print("=" * 40)

    # 1. 解析 trace
    print("\n[1/4] Parsing trace...")
    t0 = time.time()
    trace = parse_trace(trace_path, max_requests=max_requests)
    print(f"  Requests: {trace.n_requests:,}")
    print(f"  Unique objects: {trace.n_unique_objects:,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # 2. 构建流网络
    print("\n[2/4] Building flow network...")
    t0 = time.time()
    topo = build_topology(trace, cache_size)
    stats = get_topology_stats(topo)
    print(f"  Nodes: {stats.n_nodes:,}")
    print(f"  Arcs: {stats.n_inner_arcs + stats.n_outer_arcs:,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # 3. 求解
    print("\n[3/4] Solving (GPU)...")
    t0 = time.time()
    result = solve_jit(topo, max_iters=30000, tol=1e-4, step_size=0.1)
    print(f"  Iterations: {result.iterations:,}")
    print(f"  Converged: {result.converged}")
    print(f"  Primal residual: {result.primal_residual:.2e}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # 4. 计算结果
    print("\n[4/4] Computing hit ratio...")
    foo_result = process_result(trace, topo, result)

    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"  Object Hit Ratio (OHR): {foo_result.ohr*100:.2f}%")
    print(f"  Float hits: {foo_result.float_hits:,.0f}")
    print(f"  Integer hits: {foo_result.integer_hits:,}")
    print("=" * 40)

if __name__ == "__main__":
    main()
```

## 故障排除

### CUDA 错误

如果遇到 `cuSPARSE` 或 CUDA 相关错误:

```bash
# 确保使用正确的环境
source /home/ubuntu/ssd/optimalwebcaching/foo-cuopt/bin/activate

# 验证 GPU
python -c "import jax; print(jax.devices())"
```

### 内存不足

- 减少 `max_requests` 参数
- 使用更大的 `check_interval` 减少中间计算

### 收敛慢

- 增加 `max_iters` (如 50000)
- 调整 `step_size` (通常 0.05-0.2)
- 减小 `tol` 精度要求
