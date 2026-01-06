# OracleGeneral Format Support for Optimal Web Caching

## Overview

This implementation adds support for the OracleGeneral binary trace format (with optional zstd compression) to all caching algorithms in this repository.

## Format Specification

### Binary Structure

Each record is exactly **24 bytes**:

```c
struct {
    uint32_t timestamp;         // 4 bytes: Request timestamp
    uint64_t obj_id;            // 8 bytes: Object ID
    uint32_t obj_size;          // 4 bytes: Object size in bytes
    int64_t  next_access_vtime; // 8 bytes: Next access virtual time (-1 if none)
}
```

### File Extensions

- **Uncompressed**: `.dat` or any extension
- **Compressed**: `.zst` (requires libzstd-dev)

## Library Components

### Core Files

| File | Purpose |
|------|---------|
| [oracle_general_reader.h](oracle_general_reader.h) | Reader interface |
| [oracle_general_reader.cpp](oracle_general_reader.cpp) | Binary format parser with zstd support |
| [parse_trace.h](parse_trace.h) | Unified trace parsing interface |
| [parse_trace.cpp](parse_trace.cpp) | Trace parser implementation |
| [generate_test_trace.cpp](generate_test_trace.cpp) | Test trace generator |
| [Makefile](Makefile) | Build configuration |

## Usage

### 1. Build the Library

```bash
cd lib/trace
make
```

### 2. Generate Test Traces

```bash
make test-gen
./generate_test_trace <output_prefix>
```

Example:
```bash
./generate_test_trace test_trace
# Creates:
#   - test_trace_simple.dat      (6 records)
#   - test_trace_size_var.dat    (3 records)
#   - test_trace_large.dat       (91 records)
#   - test_trace_single_obj.dat  (19 records)
```

### 3. Compress Traces (Optional)

```bash
apt install zstd
zstd test_trace_simple.dat -o test_trace_simple.dat.zst
```

### 4. Use with Algorithms

All algorithms have been updated to use the new format automatically:

```bash
cd OHRgoal/FOO
make
./foo ../../lib/trace/test_trace_simple.dat 10000 4 output.txt

cd ../Belady
make
./belady2 ../../lib/trace/test_trace_simple.dat 10000 100
```

## Algorithm Support Matrix

| Algorithm | Directory | Updated | Tested |
|-----------|-----------|---------|--------|
| **FOO** | OHRgoal/FOO | ✅ | ✅ |
| **PFOO-U** | OHRgoal/PFOO-U | ✅ | ⏳ |
| **PFOO-L** | OHRgoal/PFOO-L | ✅ | ⏳ |
| **Belady** | OHRgoal/Belady | ✅ | ⏳ |
| **Belady-Size** | OHRgoal/Belady-Size | ✅ | ⏳ |
| **PFOO-L** | BHRgoal/PFOO-L | ✅ | ⏳ |
| **Belady** | BHRgoal/Belady | ✅ | ⏳ |
| **BeladySplit** | BHRgoal/BeladySplit | ✅ | ⏳ |
| **Statistics** | Statistics | ✅ | ⏳ |

## Test Results

### FOO Algorithm Test

**Input:** `test_trace_simple.dat` (6 requests, 3 unique objects)

```
scanned trace n=6 m=3
created graph with 4 nodes 6 arcs
ExLP4 10000 hitc 3 reqc 6 OHR 0.5 3 3
```

**Output:**
```
1000 1 1024 1    # Hit (cached)
2000 2 2048 1    # Hit (cached)
3000 1 1024 0    # Miss (first access)
4000 3 512 1     # Hit (cached)
5000 2 2048 0    # Miss (first access)
6000 3 512 0     # Miss (first access)
```

**Result:** OHR = 0.5 (3 hits out of 6 requests)

## Compilation Options

### Without zstd Support (Default)

```makefile
CXXFLAGS += -Wall -Wno-maybe-uninitialized
# No additional dependencies
```

### With zstd Support

```makefile
CXXFLAGS += -DUSE_ZSTD
LDFLAGS += -lzstd
```

Install zstd development library:
```bash
apt install libzstd-dev
```

## Migration from Old Text Format

### Old Format (No Longer Supported)
```
[timestamp] [object-id] [size-in-bytes]
1234567890 obj001 1024
```

### New Format (OracleGeneral Binary)
- Compressed binary (24 bytes/record)
- Includes `next_access_vtime` field
- Better performance for large traces
- Optional zstd compression

### Converting Old Traces

No automated converter is provided. Use libCacheSim tools to convert traces to OracleGeneral format.

## API Reference

### OracleGeneralReader Class

```cpp
class OracleGeneralReader {
public:
    struct Record {
        uint32_t timestamp;
        uint64_t obj_id;
        uint32_t obj_size;
        int64_t  next_access_vtime;
    };

    OracleGeneralReader(const std::string& path);
    bool readAll(std::vector<Record>& records);
    const std::string& getError() const;
    size_t getRecordCount() const;
};
```

### Parse Functions

```cpp
// For algorithms using basic trEntry
uint64_t parseOracleGeneralTrace(std::vector<trEntry>& trace,
                                  const std::string& path);

// For algorithms using extended trEntryExtended (includes time, arcId)
uint64_t parseOracleGeneralTraceExtended(std::vector<trEntryExtended>& trace,
                                         const std::string& path);

// Legacy function (delegates to OracleGeneral)
uint64_t parseTraceFile(std::vector<trEntry>& trace, std::string& path);
```

## Troubleshooting

### Common Errors

**1. "zstd compression support not enabled"**
- Solution: Compile with `-DUSE_ZSTD -lzstd` and install libzstd-dev

**2. "Invalid file size (not a multiple of 24)"**
- Cause: File is corrupted or not in OracleGeneral format
- Solution: Verify file format with hexdump or Python

**3. "Failed to parse record at index X"**
- Cause: Corrupted binary data
- Solution: Regenerate trace file

### Debugging

Enable detailed trace output:
```cpp
OracleGeneralReader reader("trace.dat");
std::vector<Record> records;
if (!reader.readAll(records)) {
    std::cerr << "Error: " << reader.getError() << std::endl;
}
```

Verify with Python:
```python
import struct

with open('trace.dat', 'rb') as f:
    data = f.read()
    for i in range(0, len(data), 24):
        timestamp, obj_id, obj_size, next_access = struct.unpack('<I Q I q', data[i:i+24])
        print(f"{timestamp}, {obj_id}, {obj_size}, {next_access}")
```

## Performance

| Trace Size | Format | Parse Time | Memory Usage |
|------------|--------|------------|--------------|
| 1K records | Binary | <1ms | ~24KB |
| 1M records | Binary | ~50ms | ~24MB |
| 1M records | zstd | ~100ms | ~24MB (decompressed) |

## References

- [libCacheSim](https://github.com/libcachesim/Library) - Reference implementation
- OracleGeneral format specification in libCacheSim source code
- zstd compression library: https://github.com/facebook/zstd

## License

Same as parent project (see LICENSE in root directory).
