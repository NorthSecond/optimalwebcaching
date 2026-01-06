# OHRgoal Module - Object Hit Ratio Optimization

[Root Directory](../CLAUDE.md) > **OHRgoal**

---

## Change Log

### 2026-01-06 21:15:22
- Initial module documentation created
- Identified 9 algorithm implementations
- Documented entry points and interfaces for all sub-modules

---

## Module Responsibilities

The **OHRgoal** module contains algorithms that optimize for **Object Hit Ratio (OHR)**, where every object counts equally regardless of size. This is appropriate for:
- **In-memory caches** (e.g., memcached)
- Systems where minimizing I/O operations is the primary goal
- Scenarios where random access latency dominates performance

### Key Concept

In OHR optimization:
- A hit for a 1GB object = 1 hit
- A hit for a 10B object = 1 hit
- Goal: Maximize total number of hits

---

## Module Structure

This module contains 9 standalone algorithm implementations:

### Network Flow Algorithms (Exact/Optimal)

| Algorithm | Description | Complexity | Status |
|-----------|-------------|------------|--------|
| **FOO** | Flow Offline Optimum - exact OPT | Very slow (network flow) | Exact but impractical |
| **PFOO-U** | Practical FOO Upper bound | Slow (iterative flow) | Upper bound on OPT |
| **PFOO-L** | Practical FOO Lower bound | Fast (greedy + flow) | Lower bound on OPT |

### Baseline and Approximation Algorithms

| Algorithm | Description | Type |
|-----------|-------------|------|
| **Belady** | Classical Belady (optimal for fixed-size) | Baseline |
| **Belady-Size** | Size-aware Belady variant | Heuristic |
| **OFMA** | Irani's approximation (STOC'97) | Theoretical approximation |
| **LocalRatio** | Local ratio approximation | Theoretical approximation |
| **Freq-Size** | Frequency-Size utility algorithm | Heuristic |
| **PFOO-U-Old** | Legacy PFOO-U implementation | Deprecated |

---

## Entry and Startup

### Build Commands

```bash
# Build specific algorithm
cd OHRgoal/[algorithm-name]
make                # Optimized build
make debug          # Debug build
make clean          # Clean artifacts

# Example: Build FOO
cd OHRgoal/FOO
make
```

### Running Algorithms

```bash
# FOO (exact OPT)
cd OHRgoal/FOO
./foo [trace.txt] [cacheSize] [pivotRule] [output.txt]
./foo trace.txt 1073741824 4 foo_output.txt

# PFOO-U (upper bound)
cd OHRgoal/PFOO-U
./pfoou [trace.txt] [cacheSize] [pivotRule] [stepSize] [output.txt]
./pfoou trace.txt 1073741824 4 500000 pfoou_output.txt

# PFOO-L (lower bound)
cd OHRgoal/PFOO-L
./pfool [trace.txt] [cacheSizeMax] [output.txt]
./pfool trace.txt 1073741824 pfool_output.txt

# Belady (baseline)
cd OHRgoal/Belady
./belady2 [trace.txt] [cacheSize] [sampleSize]
./belady2 trace.txt 1073741824 100

# OFMA (Irani's approximation)
cd OHRgoal/OFMA
./ofma [trace.txt] [cacheSize]
./ofma trace.txt 1073741824

# LocalRatio
cd OHRgoal/LocalRatio
./localratio [trace.txt] [cacheSize]
./localratio trace.txt 1073741824

# Freq-Size
cd OHRgoal/Freq-Size
./utility [trace.txt] [cacheSize]
./utility trace.txt 1073741824
```

---

## External Interfaces

### Input Format (All Algorithms)

All algorithms expect space-separated traces:
```
[timestamp] [object-id] [size-in-bytes]
1234567890 obj001 1024
1234567891 obj002 4096
```

### Output Format

**FOO**: Writes decision variables to output file
- Format: Per-request caching decisions (binary/encoded)
- Use for analysis and visualization

**PFOO algorithms**: Write decision variables to output file
- Similar to FOO but with approximated bounds

**Belady/OFMA/others**: Print hit/miss statistics to stdout
- Object hit ratio
- Object miss ratio
- Total requests, hits, misses

### Algorithm-Specific Parameters

| Algorithm | Parameters | Description |
|-----------|-----------|-------------|
| FOO | trace, cacheSize, pivotRule, output | pivotRule: LEMON NetworkSimplex pivot rule (1-7) |
| PFOO-U | trace, cacheSize, pivotRule, stepSize, output | stepSize: Decision variables per iteration (500k recommended) |
| PFOO-L | trace, cacheSizeMax, output | cacheSizeMax: Maximum cache size for sweep |
| Belady | trace, cacheSize, sampleSize | sampleSize: Number of samples for stochastic eviction |
| OFMA | trace, cacheSize | Simple 2-parameter interface |
| LocalRatio | trace, cacheSize | Simple 2-parameter interface |
| Freq-Size | trace, cacheSize | Simple 2-parameter interface |

---

## Key Dependencies and Configuration

### External Libraries

1. **LEMON Network Flow Library** (from `/lib/lemon/`)
   - Used by: FOO, PFOO-U, PFOO-L
   - Components:
     - `lemon/smart_graph.h`: Graph data structures
     - `lemon/network_simplex.h`: Min-cost flow solver
   - Integration: Compile-time include (header-only library)

2. **C++ Standard Library**
   - `<vector>`: Trace storage
   - `<unordered_map>`: Object tracking
   - `<fstream>`: File I/O
   - `<tuple>`: Trace entry data structures

### Build Configuration

All Makefiles share common flags:
```makefile
CXXFLAGS += -std=c++11
CXXFLAGS += -I.. -I ../../lib    # Include paths for LEMON
CXXFLAGS += -O3 -ffast-math -march=native
CXXFLAGS += -Wall -Werror
```

Specialized flags:
- **PFOO-L**: Additional `-funroll-loops` for performance
- **PFOO-U**: Additional `-mcmodel=medium` for large graphs
- **Debug builds**: `-ggdb -D_GLIBCXX_DEBUG`

---

## Data Models

### traceEntry Structure Variants

Each algorithm uses a different `traceEntry` structure optimized for its needs:

#### Network Flow Algorithms (FOO, PFOO-U)

```cpp
typedef std::tuple<uint64_t, uint64_t, bool, uint64_t, int> traceEntry;
// Fields: id, size, hasNext, time, arcId
```

#### PFOO-L

```cpp
struct trEntry {
    uint64_t size;           // Object size
    uint64_t volume;         // Cumulative volume
    bool hasNext;            // Has next request
    // Additional fields for algorithm
};
```

#### Belady Variants

```cpp
struct trEntry {
    const uint64_t id;       // Object ID
    const uint64_t size;     // Object size
    size_t nextSeen;         // Index of next request
    bool hasNext;            // Has future request
    bool hit;                // Was this request a hit?
};
```

#### OFMA / LocalRatio

```cpp
struct trEntry {
    const uint64_t id;
    const uint64_t size;
    double pval;             // Probability / priority value
    size_t nextSeen;
    bool inSchedule;         // In caching schedule
    bool hit;
};
```

#### Freq-Size

```cpp
struct trEntry {
    uint64_t size;
    double utility;          // Frequency-size utility
    uint64_t reqCount;       // Request frequency
};
```

### Common Operations

**parseTraceFile()**: Parse trace file into vector
```cpp
uint64_t parseTraceFile(std::vector<traceEntry> & trace, std::string & path);
```

**createMCF()**: Build min-cost flow graph (network flow algorithms)
```cpp
void createMCF(SmartDigraph & g,
               std::vector<traceEntry> & trace,
               uint64_t cacheSize,
               SmartDigraph::ArcMap<int64_t> & cap,
               SmartDigraph::ArcMap<double> & cost,
               SmartDigraph::NodeMap<int64_t> & supplies);
```

**solveMCF()**: Solve optimization using LEMON
```cpp
double solveMCF(SmartDigraph & g,
                SmartDigraph::ArcMap<int64_t> & cap,
                SmartDigraph::ArcMap<double> & cost,
                SmartDigraph::NodeMap<int64_t> & supplies,
                SmartDigraph::ArcMap<uint64_t> & flow,
                int solverPar);
```

**cacheAlg()**: Execute caching algorithm (Belady variants, OFMA, etc.)
```cpp
void cacheAlg(std::vector<trEntry> & trace, uint64_t cacheSize, ...);
```

**printRes()**: Output results
```cpp
void printRes(std::vector<trEntry> & trace, std::string algName, ...);
```

---

## Testing and Quality

### Current Test Status
- **No module-specific tests exist**
- Tests only exist at repository root level (`/tests/test_createMCF.cpp`)
- Test coverage is minimal (early-stage research prototype)

### Manual Testing Approach

```bash
# Test with small trace
cd OHRgoal/FOO
make
./foo test.tr 1024 4 output.txt

# Compare algorithms
cd ../Belady
make
./belady2 test.tr 1024 100

cd ../OFMA
make
./ofma test.tr 1024
```

### Expected Output Validation
- Hit ratio should be between 0.0 and 1.0
- OPT (FOO) should have highest hit ratio
- PFOO-U >= OPT >= PFOO-L (upper and lower bounds)
- Belady should have reasonable hit ratio (baseline)

---

## Algorithm-Specific Notes

### FOO (Flow Offline Optimum)
- **Purpose**: Calculate exact optimal caching policy
- **Method**: Formulate as min-cost flow problem, solve with NetworkSimplex
- **Pros**: Asymptotically exact
- **Cons**: Very slow, does not scale to large traces
- **Use Case**: Ground truth for small traces, validation

### PFOO-U (Practical FOO Upper Bound)
- **Purpose**: Fast upper bound on OPT
- **Method**: Iterative network flow with fixed step size
- **Parameters**:
  - `stepSize`: Number of decision variables per iteration
  - Lower = faster, Higher = more accurate
  - Recommended: 500000
- **Pros**: More scalable than FOO
- **Cons**: Still slow for large traces

### PFOO-L (Practical FOO Lower Bound)
- **Purpose**: Fast lower bound on OPT
- **Method**: Greedy algorithm + network flow refinement
- **Pros**: Fastest network flow algorithm
- **Cons**: Lower bound only (not optimal)

### Belady
- **Purpose**: Baseline comparison
- **Method**: Evict object with farthest next request
- **Assumption**: Future knowledge of requests
- **Note**: Optimal for fixed-size objects, suboptimal for variable sizes

### Belady-Size
- **Purpose**: Size-aware Belady variant
- **Method**: Consider both next request time and object size
- **Improvement**: Better than Belady for variable sizes

### OFMA (Irani's Algorithm)
- **Purpose**: Theoretical approximation from STOC 1997
- **Method**: Utility-based with probabilistic framework
- **Reference**: "Page replacement with multi-size pages and applications to web caching"

### LocalRatio
- **Purpose**: Local ratio approximation technique
- **Method**: Iterative local optimization
- **Reference**: "A unified approach to approximating resource allocation and scheduling" (J. ACM 2001)

### Freq-Size
- **Purpose**: Frequency-Size utility heuristic
- **Method**: Cache objects with high frequency-to-size ratio
- **Intuition**: Value = requests per byte

### PFOO-U-Old (Deprecated)
- **Status**: Legacy implementation
- **Superseded by**: PFOO-U
- **Reason for deprecation**: Performance improvements in newer version

---

## Frequently Asked Questions (FAQ)

**Q: Which algorithm should I use?**
A:
- For exact OPT on small traces: FOO
- For bounds on medium traces: PFOO-U (upper) + PFOO-L (lower)
- For fast baseline: Belady
- For large traces: Belady-Size or Freq-Size

**Q: What is the "pivot rule" parameter in FOO/PFOO-U?**
A: This is the LEMON NetworkSimplex pivot rule (1-7). Rule 4 is commonly used. Different rules affect convergence speed but not optimality.

**Q: How do I interpret the output?**
A:
- Decision variable files (FOO/PFOO): Binary format for analysis
- Stdout (Belady/OFMA): Human-readable hit/miss statistics
- Look for "OHR" or "object hit ratio" in output

**Q: Why is PFOO-U so slow?**
A: Known issue (GitHub Issue #1). Network simplex solver is not optimally configured. Contributions welcome.

**Q: Can I use these algorithms in production?**
A: These are research prototypes. They require offline trace processing and are not designed for real-time caching systems.

**Q: What trace size can these algorithms handle?**
A:
- FOO: Up to ~10k requests (very slow beyond)
- PFOO-U: Up to ~100k requests (with large step size)
- PFOO-L: Up to ~1M requests (depends on memory)
- Belady variants: Unlimited (very fast)

---

## Related File List

### Network Flow Algorithms
- `/OHRgoal/FOO/foo.cpp` - FOO entry point
- `/OHRgoal/FOO/lib/solve_mcf.cpp` - Network flow solver
- `/OHRgoal/FOO/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/FOO/lib/solve_mcf.h` - Solver interface
- `/OHRgoal/FOO/lib/parse_trace.h` - Parser interface
- `/OHRgoal/FOO/Makefile` - Build configuration

- `/OHRgoal/PFOO-U/pfoou.cpp` - PFOO-U entry point
- `/OHRgoal/PFOO-U/lib/solve_mcf.cpp` - Network flow solver
- `/OHRgoal/PFOO-U/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/PFOO-U/lib/solve_mcf.h` - Solver interface
- `/OHRgoal/PFOO-U/lib/parse_trace.h` - Parser interface
- `/OHRgoal/PFOO-U/Makefile` - Build configuration

- `/OHRgoal/PFOO-L/pfool.cpp` - PFOO-L entry point
- `/OHRgoal/PFOO-L/lib/solve_mcf.cpp` - Network flow solver
- `/OHRgoal/PFOO-L/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/PFOO-L/lib/solve_mcf.h` - Solver interface
- `/OHRgoal/PFOO-L/lib/parse_trace.h` - Parser interface
- `/OHRgoal/PFOO-L/Makefile` - Build configuration

### Baseline Algorithms
- `/OHRgoal/Belady/belady2.cpp` - Belady entry point
- `/OHRgoal/Belady/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/Belady/lib/solve_mcf.cpp` - Network flow (for comparison)
- `/OHRgoal/Belady/lib/parse_trace.h` - Parser interface
- `/OHRgoal/Belady/lib/solve_mcf.h` - Solver interface
- `/OHRgoal/Belady/Makefile` - Build configuration

- `/OHRgoal/Belady-Size/belady2size.cpp` - Belady-Size entry point
- `/OHRgoal/Belady-Size/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/Belady-Size/lib/solve_mcf.cpp` - Network flow
- `/OHRgoal/Belady-Size/lib/parse_trace.h` - Parser interface
- `/OHRgoal/Belady-Size/lib/solve_mcf.h` - Solver interface
- `/OHRgoal/Belady-Size/Makefile` - Build configuration

### Approximation Algorithms
- `/OHRgoal/OFMA/ofma.cpp` - OFMA entry point (standalone)
- `/OHRgoal/OFMA/Makefile` - Build configuration

- `/OHRgoal/LocalRatio/localratio.cpp` - LocalRatio entry point (standalone)
- `/OHRgoal/LocalRatio/Makefile` - Build configuration

- `/OHRgoal/Freq-Size/utility.cpp` - Freq-Size entry point
- `/OHRgoal/Freq-Size/lib/parse_trace.cpp` - Trace parser
- `/OHRgoal/Freq-Size/lib/parse_trace.h` - Parser interface
- `/OHRgoal/Freq-Size/Makefile` - Build configuration

### Legacy
- `/OHRgoal/PFOO-U-Old/pfoou.cpp` - Legacy PFOO-U (deprecated)
- `/OHRgoal/PFOO-U-Old/lib/parse_trace.cpp`
- `/OHRgoal/PFOO-U-Old/lib/solve_mcf.cpp`
- `/OHRgoal/PFOO-U-Old/lib/parse_trace.h`
- `/OHRgoal/PFOO-U-Old/lib/solve_mcf.h`
- `/OHRgoal/PFOO-U-Old/Makefile`

---

## See Also

- **Root Documentation**: [CLAUDE.md](../CLAUDE.md)
- **BHRgoal Module**: [BHRgoal/CLAUDE.md](../BHRgoal/CLAUDE.md) - Byte Hit Ratio optimization
- **Research Paper**: [SIGMETRICS 2018](https://www.cs.cmu.edu/~dberger1/pdf/2018PracticalBound_SIGMETRICS.pdf)
- **Trace Format**: [webcachesim README](https://github.com/dasebe/webcachesim/edit/master/README.md)
