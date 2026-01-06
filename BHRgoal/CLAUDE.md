# BHRgoal Module - Byte Hit Ratio Optimization

[Root Directory](../CLAUDE.md) > **BHRgoal**

---

## Change Log

### 2026-01-06 21:15:22
- Initial module documentation created
- Identified 3 algorithm implementations
- Documented entry points and interfaces for all sub-modules

---

## Module Responsibilities

The **BHRgoal** module contains algorithms that optimize for **Byte Hit Ratio (BHR)**, where objects are weighted in proportion to their size. This is appropriate for:
- **Disk/flash caches** (e.g., CDNs like Varnish)
- Systems where bandwidth cost is the primary concern
- Scenarios where each miss incurs a bandwidth cost linear in missed bytes

### Key Concept

In BHR optimization:
- A hit for a 4KB object = 4 KB saved
- A hit for a 1KB object = 1 KB saved
- A hit for a 4KB object is worth 4x more than a 1KB hit
- Goal: Maximize total bytes served from cache

---

## Module Structure

This module contains 3 algorithm implementations:

| Algorithm | Description | Type | Status |
|-----------|-------------|------|--------|
| **PFOO-L** | Practical FOO Lower bound | Network flow + greedy | Production-ready |
| **Belady** | Belady algorithm for BHR | Greedy baseline | Production-ready |
| **BeladySplit** | Split-object Belady variant | Greedy heuristic | Experimental |

### Missing Implementations

- **PFOO-U** (Upper bound): Listed in README but not yet ported from OHR version
  - Status: TODO / In Progress
  - Note: Will provide upper bound on BHR when completed

---

## Entry and Startup

### Build Commands

```bash
# Build specific algorithm
cd BHRgoal/[algorithm-name]
make                # Optimized build
make debug          # Debug build
make clean          # Clean artifacts

# Example: Build PFOO-L
cd BHRgoal/PFOO-L
make
```

### Running Algorithms

```bash
# PFOO-L (lower bound on byte miss ratio)
cd BHRgoal/PFOO-L
./pfool [trace.txt] [cacheSize]
./pfool trace.txt 1073741824

# Output:
# OHR: 0.XXXX
# BHR: 0.XXXX

# Belady (baseline for BHR)
cd BHRgoal/Belady
./belady2 [trace.txt] [cacheSize] [sampleSize]
./belady2 trace.txt 1073741824 100

# Output:
# OHR: 0.XXXX
# BHR: 0.XXXX

# BeladySplit (split-object variant)
cd BHRgoal/BeladySplit
./belady2 [trace.txt] [cacheSize] [sampleSize]
./belady2 trace.txt 1073741824 100

# Output:
# OHR: 0.XXXX
# BHR: 0.XXXX
```

---

## External Interfaces

### Input Format (All Algorithms)

All algorithms expect space-separated traces:
```
[timestamp] [object-id] [size-in-bytes]
1234567890 obj001 1024
1234567891 obj002 4096
1234567892 obj001 1024
```

### Output Format

All BHR algorithms output **both** Object Hit Ratio (OHR) and Byte Hit Ratio (BHR) to stdout:

```
OHR: 0.7534
BHR: 0.8126
```

Where:
- **OHR** = (object hits) / (total requests)
- **BHR** = (bytes served from cache) / (total bytes requested)

### Algorithm-Specific Parameters

| Algorithm | Parameters | Description |
|-----------|-----------|-------------|
| PFOO-L | trace, cacheSize | Simple 2-parameter interface |
| Belady | trace, cacheSize, sampleSize | sampleSize: Stochastic eviction samples |
| BeladySplit | trace, cacheSize, sampleSize | sampleSize: Stochastic eviction samples |

---

## Key Dependencies and Configuration

### External Libraries

1. **LEMON Network Flow Library** (from `/lib/lemon/`)
   - Used by: PFOO-L
   - Components:
     - `lemon/smart_graph.h`: Graph data structures
     - `lemon/network_simplex.h`: Min-cost flow solver
   - Integration: Compile-time include (header-only library)

2. **C++ Standard Library**
   - `<vector>`: Trace storage
   - `<unordered_map>`: Object tracking
   - `<fstream>`: File I/O
   - `<set>`: Ordered object tracking (Belady variants)

### Build Configuration

All Makefiles share common flags:
```makefile
CXXFLAGS += -std=c++11
CXXFLAGS += -I.. -I ../../lib    # Include paths for LEMON
CXXFLAGS += -O3 -ffast-math -march=native
CXXFLAGS += -funroll-loops        # Additional loop unrolling
CXXFLAGS += -Wall -Werror
```

Specialized flags:
- **PFOO-L**: Additional `-funroll-loops` for performance
- **Debug builds**: `-ggdb -D_GLIBCXX_DEBUG`

---

## Data Models

### traceEntry Structure (BHR-specific)

#### PFOO-L

```cpp
struct trEntry {
    uint64_t size;           // Object size in bytes
    uint64_t volume;         // Cumulative volume (for algorithm)
    bool hasNext;            // Has next request

    trEntry(uint64_t nsize)
        : size(nsize), volume(0), hasNext(false) {}
};
```

#### Belady / BeladySplit

```cpp
struct trEntry {
    const uint64_t id;       // Object ID
    const uint64_t size;     // Object size in bytes
    size_t nextSeen;         // Index of next request
    bool hasNext;            // Has future request
    double hit;              // Hit fraction (for BHR calculation)

    trEntry(uint64_t nid, uint64_t nsize)
        : id(nid), size(nsize), nextSeen(0),
          hasNext(false), hit(0.0) {}
};
```

### Common Operations

**parseTraceFile()**: Parse trace file into vector
```cpp
// PFOO-L variant
void parseTraceFile(std::vector<trEntry> & trace,
                    std::string & path,
                    uint64_t & byteSum);

// Belady variant
uint64_t parseTraceFile(std::vector<trEntry> & trace,
                        std::string & path);
```

**createMCF()**: Build min-cost flow graph (PFOO-L only)
```cpp
void createMCF(SmartDigraph & g,
               std::vector<trEntry> & trace,
               uint64_t cacheSize,
               SmartDigraph::ArcMap<int64_t> & cap,
               SmartDigraph::ArcMap<double> & cost,
               SmartDigraph::NodeMap<int64_t> & supplies);
```

**solveMCF()**: Solve optimization using LEMON (PFOO-L only)
```cpp
void solveMCF(SmartDigraph & g,
              SmartDigraph::ArcMap<int64_t> & cap,
              SmartDigraph::ArcMap<double> & cost,
              SmartDigraph::NodeMap<int64_t> & supplies);
```

**cacheAlg()**: Execute caching algorithm
```cpp
// PFOO-L
void cacheAlg(std::vector<trEntry> & trace);

// Belady variants
void cacheAlg(std::vector<trEntry> & trace,
              uint64_t cacheSize,
              size_t sampleSize);
```

**printRes()**: Output OHR and BHR results
```cpp
// PFOO-L
void printRes(std::vector<trEntry> & trace,
              uint64_t byteSum,
              uint64_t cacheSize);

// Belady variants
void printRes(std::vector<trEntry> & trace,
              std::string algName,
              uint64_t cacheSize,
              std::ofstream * r);
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
cd BHRgoal/PFOO-L
make
./pfool test.tr 1024

# Compare with baseline
cd ../Belady
make
./belady2 test.tr 1024 100

# Compare split-object variant
cd ../BeladySplit
make
./belady2 test.tr 1024 100
```

### Expected Output Validation
- OHR and BHR should both be between 0.0 and 1.0
- BHR typically higher than OHR (large objects contribute more)
- PFOO-L should provide **lower bound** on BHR (upper bound on byte miss ratio)
- Belady should provide reasonable baseline performance
- For same trace, OHR+BHR should be consistent across runs

---

## Algorithm-Specific Notes

### PFOO-L (Practical FOO Lower Bound)

**Purpose**: Fast lower bound on byte miss ratio (i.e., upper bound on BHR)

**Method**:
1. Greedy interval partitioning
2. Min-cost flow optimization
3. Byte-weighted objective function

**Characteristics**:
- **Output**: Lower bound on OPT's byte miss ratio
- **Interpretation**: No online policy can achieve higher BHR than this
- **Speed**: Fast (scales to large traces)
- **Use Case**: Benchmark ceiling for BHR optimization

**Key Differences from OHR PFOO-L**:
- Uses byte-weighted costs instead of unit costs
- Optimizes for total bytes cached, not object count
- Output includes both OHR and BHR for comparison

### Belady

**Purpose**: Baseline BHR algorithm

**Method**:
- Classical Belady with byte-weighted eviction
- Evict object with farthest next request
- In case of ties, may consider object size

**Characteristics**:
- **Offline**: Requires knowledge of future requests
- **Not optimal**: Belady is not optimal for variable-size objects with BHR
- **Use Case**: Performance baseline

**Sampling**:
- `sampleSize` parameter for stochastic eviction
- Higher sample size = more deterministic, slower
- Recommended: 100 for most cases

### BeladySplit (Experimental)

**Purpose**: Split-object Belady variant

**Method**:
- Allows splitting large objects into smaller chunks
- May improve BHR by better utilizing cache space
- Experimental approach

**Characteristics**:
- **Status**: Experimental / Research prototype
- **Use Case**: Investigate if splitting improves BHR
- **Note**: Not production-ready

---

## OHR vs BHR: Key Differences

### Optimization Goal

| Metric | What it Optimizes | Use Case |
|--------|-------------------|----------|
| **OHR** | Minimize number of cache misses | In-memory caches (memcached), I/O operation minimization |
| **BHR** | Minimize bytes transferred | CDN caches (Varnish), bandwidth minimization |

### Example

Consider a cache with 2 requests:
- Request 1: Object A (1 GB)
- Request 2: Object B (1 KB), repeated 1000 times

**OHR-optimal policy**: Cache object B
- Hits: 999/1000 = 99.9% hit ratio
- Bytes served: 999 KB / (1 GB + 1 MB) ≈ 0.1% BHR

**BHR-optimal policy**: Cache object A
- Hits: 0/1001 = 0% hit ratio
- Bytes served: 1 GB / (1 GB + 1 MB) ≈ 99.9% BHR

This extreme example shows how OHR and BHR can lead to drastically different policies!

---

## Frequently Asked Questions (FAQ)

**Q: Which BHR algorithm should I use?**
A:
- For benchmark upper bound: PFOO-L
- For fast baseline: Belady
- For experimentation: BeladySplit

**Q: Why is there no BHR PFOO-U?**
A: It hasn't been ported from the OHR version yet. This is listed as a TODO in the README.

**Q: How do I interpret OHR vs BHR output?**
A:
- OHR: Fraction of requests that hit (0.0 to 1.0)
- BHR: Fraction of bytes served from cache (0.0 to 1.0)
- For CDNs, BHR is usually more important (bandwidth cost)

**Q: Can BHR be lower than OHR?**
A: Yes, typically BHR > OHR because large objects contribute more bytes. But if large objects are never cached, BHR could be lower.

**Q: Why does Belady need a sample size parameter?**
A: For stochastic eviction when multiple objects have the same next-request distance. Higher = more deterministic.

**Q: What's the difference between Belady and BeladySplit?**
A: BeladySplit allows splitting large objects into smaller chunks, potentially improving BHR by better cache space utilization.

**Q: Can I use these for real-time caching?**
A: No, these are offline analysis tools. Real-time caching would use online policies like LRU, AdaptSize, GDSF.

---

## Related File List

### PFOO-L
- `/BHRgoal/PFOO-L/pfool.cpp` - PFOO-L entry point
- `/BHRgoal/PFOO-L/lib/solve_mcf.cpp` - Network flow solver
- `/BHRgoal/PFOO-L/lib/solve_mcf_log.cpp` - Logging utilities
- `/BHRgoal/PFOO-L/lib/parse_trace.cpp` - Trace parser
- `/BHRgoal/PFOO-L/lib/solve_mcf.h` - Solver interface
- `/BHRgoal/PFOO-L/lib/parse_trace.h` - Parser interface
- `/BHRgoal/PFOO-L/Makefile` - Build configuration

### Belady
- `/BHRgoal/Belady/belady2.cpp` - Belady entry point
- `/BHRgoal/Belady/lib/parse_trace.cpp` - Trace parser
- `/BHRgoal/Belady/lib/solve_mcf.cpp` - Network flow (for comparison)
- `/BHRgoal/Belady/lib/parse_trace.h` - Parser interface
- `/BHRgoal/Belady/lib/solve_mcf.h` - Solver interface
- `/BHRgoal/Belady/Makefile` - Build configuration

### BeladySplit
- `/BHRgoal/BeladySplit/belady2.cpp` - BeladySplit entry point
- `/BHRgoal/BeladySplit/lib/parse_trace.cpp` - Trace parser
- `/BHRgoal/BeladySplit/lib/solve_mcf.cpp` - Network flow
- `/BHRgoal/BeladySplit/lib/parse_trace.h` - Parser interface
- `/BHRgoal/BeladySplit/lib/solve_mcf.h` - Solver interface
- `/BHRgoal/BeladySplit/Makefile` - Build configuration

---

## See Also

- **Root Documentation**: [CLAUDE.md](../CLAUDE.md)
- **OHRgoal Module**: [OHRgoal/CLAUDE.md](../OHRgoal/CLAUDE.md) - Object Hit Ratio optimization
- **Research Paper**: [SIGMETRICS 2018](https://www.cs.cmu.edu/~dberger1/pdf/2018PracticalBound_SIGMETRICS.pdf)
- **Trace Format**: [webcachesim README](https://github.com/dasebe/webcachesim/edit/master/README.md)

---

## Future Work

### Known TODOs

1. **Port PFOO-U from OHR to BHR**
   - Status: Listed in README, not implemented
   - Complexity: Network flow with byte-weighted costs
   - Priority: High (completes upper/lower bound pair)

2. **Add more BHR optimization algorithms**
   - Current: Only PFOO-L, Belady, BeladySplit
   - Potential: Size-aware LRU variants, utility-based algorithms

3. **Improve test coverage**
   - No BHR-specific tests exist
   - Need validation of BHR calculations

4. **Performance optimization**
   - PFOO-L could benefit from algorithmic improvements
   - Consider parallelization for large traces
