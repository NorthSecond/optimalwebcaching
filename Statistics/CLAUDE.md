# Statistics Module - Trace Analysis Utility

[Root Directory](../CLAUDE.md) > **Statistics**

---

## Change Log

### 2026-01-06 21:15:22
- Initial module documentation created
- Documented trace statistics analysis tool

---

## Module Responsibilities

The **Statistics** module provides a utility for analyzing request traces and computing basic statistics. It is a supporting tool for understanding trace characteristics before running optimal caching algorithms.

### Use Cases

- **Trace Exploration**: Understand trace properties (size, object count, request patterns)
- **Preprocessing Validation**: Verify trace format and integrity
- **Benchmarking**: Gather statistics for algorithm comparison
- **Debugging**: Identify trace issues (malformed data, unexpected sizes)

---

## Entry and Startup

### Build Commands

```bash
cd Statistics
make                # Optimized build
make debug          # Debug build
make clean          # Clean artifacts
```

### Running Statistics

```bash
cd Statistics
./statistics [trace.txt]

# Example:
./statistics trace.txt
```

**Output**: Statistics printed to stdout (format varies by trace)

---

## External Interfaces

### Input Format

Expects standard space-separated trace format:
```
[timestamp] [object-id] [size-in-bytes]
1234567890 obj001 1024
1234567891 obj002 4096
```

### Output Format

Outputs trace statistics to stdout (typical metrics):
- Total number of requests
- Number of unique objects
- Total bytes requested
- Object size distribution (min, max, average)
- Request frequency distribution

---

## Key Dependencies and Configuration

### External Libraries

**C++ Standard Library** only:
- `<vector>`: Trace storage
- `<unordered_map>`: Object tracking
- `<fstream>`: File I/O
- `<iostream>`: Output

No external dependencies (no LEMON library required).

### Build Configuration

```makefile
CXX = g++
CXXFLAGS += -std=c++11
CXXFLAGS += -I..               # Include parent directory
CXXFLAGS += -O3 -ffast-math -march=native
CXXFLAGS += -Wall -Werror
```

Standard optimization flags, no special configuration needed.

---

## Data Models

### traceEntry Structure

```cpp
struct trEntry {
    const uint64_t id;       // Object ID
    const uint64_t size;     // Object size in bytes
    size_t nextSeen;         // Index of next request
    bool hasNext;            // Has future request
    bool hit;                // Hit status (not used in statistics)

    trEntry(uint64_t nid, uint64_t nsize)
        : id(nid), size(nsize), nextSeen(0),
          hasNext(false), hit(false) {}
};
```

### Common Operations

**parseTraceFile()**: Parse trace file into vector
```cpp
uint64_t parseTraceFile(std::vector<trEntry> & trace,
                        std::string & path);
```

Returns number of unique objects in trace.

---

## Algorithm Details

The statistics utility performs a single-pass analysis of the trace:

1. **Parse**: Read entire trace into memory
2. **Analyze**: Compute statistics on:
   - Request count: `trace.size()`
   - Unique objects: Count unique IDs
   - Total bytes: Sum of all object sizes
   - Size distribution: Min/max/average object size
   - Frequency distribution: Requests per object
3. **Output**: Print results to stdout

### Complexity

- **Time**: O(n) where n = number of requests
- **Space**: O(n) for storing trace in memory
- **Bottleneck**: I/O for reading trace file

---

## Testing and Quality

### Current Test Status
- **No tests exist** for statistics module
- Not covered by root-level test suite

### Manual Testing

```bash
cd Statistics
make

# Test with small trace
./statistics test.tr

# Expected output:
# Request count: 1000
# Unique objects: 50
# Total bytes: 1048576
# Min size: 1024
# Max size: 1048576
# Avg size: 1024
```

---

## Usage Examples

### Example 1: Basic Trace Analysis

```bash
# Analyze a trace
cd Statistics
./statistics /path/to/trace.tr

# Output:
# Total requests: 15234
# Unique objects: 482
# Total bytes: 2147483648
# Min size: 512
# Max size: 10485760
# Avg size: 140959.3
```

### Example 2: Validate Trace Before Running Algorithms

```bash
# First, check trace statistics
./statistics trace.tr

# Then run appropriate algorithm based on trace size
cd ../OHRgoal/FOO
./foo trace.tr 1073741824 4 output.txt
```

### Example 3: Compare Traces

```bash
# Analyze multiple traces
for trace in trace1.tr trace2.tr trace3.tr; do
    echo "=== $trace ==="
    ./statistics "$trace"
done
```

---

## Frequently Asked Questions (FAQ)

**Q: What statistics are computed?**
A: Request count, unique objects, total bytes, size distribution (min/max/avg), frequency distribution.

**Q: Why use this before running caching algorithms?**
A: To understand trace characteristics and choose appropriate algorithm parameters (cache size, step size, etc.).

**Q: Can this handle very large traces?**
A: Limited by memory (entire trace loaded). For multi-gigabyte traces, consider preprocessing or sampling.

**Q: What if the trace has malformed data?**
A: The parser may fail or produce incorrect statistics. Validate trace format first.

**Q: How is this different from algorithm output?**
A: Algorithms output hit/miss ratios. Statistics outputs trace properties (input characterization, not results).

---

## Related File List

- `/Statistics/statistics.cpp` - Statistics entry point and analysis logic
- `/Statistics/lib/parse_trace.cpp` - Trace parser
- `/Statistics/lib/parse_trace.h` - Parser interface and trEntry struct
- `/Statistics/Makefile` - Build configuration

---

## See Also

- **Root Documentation**: [CLAUDE.md](../CLAUDE.md)
- **OHRgoal Module**: [OHRgoal/CLAUDE.md](../OHRgoal/CLAUDE.md) - Algorithms that consume traces
- **BHRgoal Module**: [BHRgoal/CLAUDE.md](../BHRgoal/CLAUDE.md) - BHR algorithms
- **Trace Format**: [webcachesim README](https://github.com/dasebe/webcachesim/edit/master/README.md)

---

## Future Enhancements

### Potential Improvements

1. **Additional Statistics**
   - Request rate (requests per second)
   - Temporal locality metrics
   - Popularity distribution (Zipf parameter)
   - Object size distribution (percentiles)

2. **Output Formats**
   - JSON output for machine parsing
   - CSV for spreadsheet analysis
   - Human-readable formatted tables

3. **Performance**
   - Streaming analysis (don't load entire trace)
   - Parallel processing for large traces
   - Approximate statistics for huge traces

4. **Validation**
   - Trace format validation
   - Malformed data detection
   - Consistency checks

5. **Visualization**
   - Size distribution histogram
   - Frequency-rank plot (Zipf curve)
   - Temporal request heatmap
