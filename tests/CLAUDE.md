# Tests Module - Unit Testing

[Root Directory](../CLAUDE.md) > **tests**

---

## Change Log

### 2026-01-06 21:15:22
- Initial module documentation created
- Documented test infrastructure and existing tests

---

## Module Responsibilities

The **tests** module provides unit testing infrastructure using the [Catch2](https://github.com/catchorg/Catch2) testing framework. It validates core functionality, particularly the Min-Cost Flow (MCF) graph creation used by network flow algorithms.

### Current Status

- **Framework**: Catch2 (header-only)
- **Coverage**: Minimal (early-stage research prototype)
- **Priority**: High (test contributions explicitly requested in README)

---

## Entry and Startup

### Build Commands

```bash
cd tests
make                # Build test executable
make clean          # Clean build artifacts
```

### Running Tests

```bash
cd tests
make test          # Build and run all tests
# OR
./runtests         # Run tests directly
```

### Expected Output

Catch2 provides formatted test output:
```
===============================================================================
All tests passed (X assertions in Y test cases)
```

---

## External Interfaces

### Test Framework

**Catch2**: Header-only C++ test framework
- **Version**: Catch2 (not specified, likely v1.x or v2.x)
- **Integration**: Included as header (likely in tests directory or system)
- **License**: Boost Software License 1.0

### Test Structure

```cpp
#define CATCH_CONFIG_MAIN  // Provides main()
#include "catch.hpp"

// Test cases...
```

---

## Key Dependencies and Configuration

### External Libraries

1. **Catch2 Testing Framework**
   - Location: Likely in `/tests/` directory (not visible in file list)
   - License: Boost Software License 1.0
   - Usage: Header-only (no linking required)

2. **Parent Module Code**
   - Uses `/lib/parse_trace.cpp` from parent directory
   - Tests network flow graph creation

### Build Configuration

```makefile
TARGET = runtests
OBJS += test_createMCF.o
OBJS += ../lib/parse_trace.o
OBJS += main.o

CXX = g++
CXXFLAGS += -std=c++11
CXXFLAGS += -I../           # Include parent directory
CXXFLAGS += -Wall -Werror
```

No optimization flags (tests prioritize correctness over speed).

---

## Data Models

### Trace Parsing (from parent module)

Uses shared `parseTraceFile()` function:
```cpp
uint64_t parseTraceFile(std::vector<traceEntry> & trace,
                        std::string & path);
```

### Network Flow Graph

Tests use LEMON graph structures:
```cpp
SmartDigraph g;                                    // Graph
SmartDigraph::ArcMap<int64_t> cap(g);             // Capacities
SmartDigraph::ArcMap<double> cost(g);             // Costs
SmartDigraph::NodeMap<int64_t> supplies(g);       // Supplies
```

---

## Testing Coverage

### Current Test Cases

Based on file analysis, only one test file exists:

#### test_createMCF.cpp

Tests **Min-Cost Flow graph creation**:
- Validates correct graph construction for caching problem
- Ensures node/edge connectivity
- Verifies capacity and cost assignments
- Likely tests small synthetic traces

**What it probably tests**:
- Correct number of nodes (requests + source + sink)
- Correct arc capacities (cache capacity, object sizes)
- Correct costs (hit/miss penalties)
- Flow conservation at nodes

### Missing Test Coverage

**High Priority** (explicitly requested in README):
- Algorithm-specific tests (FOO, PFOO, Belady variants)
- Trace parsing validation
- Hit/miss ratio calculations
- Edge cases (empty cache, single object, etc.)

**Medium Priority**:
- Performance regression tests
- Memory leak detection
- Large trace handling

**Low Priority**:
- Randomized testing
- Property-based testing
- Fuzzing

---

## Test Examples

### Example Test Case (Hypothetical)

```cpp
TEST_CASE("MCF graph creation", "[mcf]") {
    // Create synthetic trace
    std::vector<traceEntry> trace;
    trace.push_back(traceEntry(1, 1024, true, 0, -1));
    trace.push_back(traceEntry(2, 2048, false, 1, -1));

    // Build graph
    SmartDigraph g;
    SmartDigraph::ArcMap<int64_t> cap(g);
    SmartDigraph::ArcMap<double> cost(g);
    SmartDigraph::NodeMap<int64_t> supplies(g);

    createMCF(g, trace, 4096, cap, cost, supplies);

    // Verify structure
    int nodeCount = 0;
    for (SmartDigraph::NodeIt n(g); n != INVALID; ++n) {
        ++nodeCount;
    }

    REQUIRE(nodeCount == 4);  // 2 requests + source + sink
}
```

---

## Running Specific Tests

### Run All Tests

```bash
cd tests
./runtests
```

### Run Specific Test Cases

```bash
# Run tests matching tag
./runtests "[mcf]"

# Run test by name
./runtests "test name"
```

### Verbose Output

```bash
./runtests -r console    # Detailed console output
./runtests -r compact    # Compact output
./runtests -s            # Show all sections
```

---

## Adding New Tests

### Template for New Test

```cpp
// In test_createMCF.cpp or new file
TEST_CASE("Test description", "[tag]") {
    // Arrange
    std::vector<traceEntry> trace;
    // ... setup test data

    // Act
    // ... call function under test

    // Assert
    REQUIRE(condition);        // Fatal failure
    CHECK(condition);          // Non-fatal (continues)
    REQUIRE_THAT(value, matcher);  // Advanced matching
}
```

### Recommended Test Additions

**High Priority**:
1. **Trace Parsing Tests**
   - Valid trace format
   - Malformed trace (missing fields)
   - Empty trace
   - Single request
   - Large trace (stress test)

2. **Algorithm Tests**
   - FOO: Small trace with known optimal solution
   - PFOO-L: Verify bounds (PFOO-L ≤ OPT ≤ PFOO-U)
   - Belady: Compare against brute-force for tiny traces
   - Hit ratio calculations (OHR and BHR)

3. **Edge Cases**
   - Zero cache size
   - Cache larger than total working set
   - All objects same size (Belady should be optimal)
   - Single object repeated

**Example Test Structure**:
```bash
tests/
├── main.cpp                  # Catch2 main
├── test_createMCF.cpp         # Existing (MCF graph creation)
├── test_parse_trace.cpp       # NEW: Trace parsing
├── test_foo.cpp              # NEW: FOO algorithm
├── test_pfoo.cpp             # NEW: PFOO algorithms
├── test_belady.cpp           # NEW: Belady variants
└── test_integration.cpp      # NEW: End-to-end tests
```

---

## Frequently Asked Questions (FAQ)

**Q: Why is test coverage so low?**
A: This is an early-stage research prototype. Test contributions are prioritized for merge (see README).

**Q: How do I debug test failures?**
A:
1. Run with `-r console -s` for detailed output
2. Use `make debug` to build with debug symbols
3. Use `REQUIRE` for fatal failures, `CHECK` for non-fatal

**Q: Can I use GDB with tests?**
A: Yes. Build with `make debug`, then `gdb ./runtests`.

**Q: How do I test specific algorithms?**
A: Create small synthetic traces with known optimal solutions, compare against algorithm output.

**Q: Where do I put new test files?**
A: Add `.cpp` files to `/tests/`, update `Makefile` OBJS, include in compilation.

**Q: Are there performance benchmarks?**
A: No, only unit tests. Performance testing is manual (see algorithm directories).

---

## Related File List

- `/tests/main.cpp` - Catch2 test runner (provides main())
- `/tests/test_createMCF.cpp` - Existing tests for MCF graph creation
- `/tests/Makefile` - Build configuration
- `/tests/LICENSE.txt` - Catch2 license (Boost Software License 1.0)
- `/lib/parse_trace.cpp` - Shared code (used by tests)
- `/lib/parse_trace.h` - Shared headers (used by tests)

---

## See Also

- **Root Documentation**: [CLAUDE.md](../CLAUDE.md)
- **Catch2 Documentation**: [https://github.com/catchorg/Catch2](https://github.com/catchorg/Catch2)
- **OHRgoal Module**: [OHRgoal/CLAUDE.md](../OHRgoal/CLAUDE.md) - Algorithms to test
- **BHRgoal Module**: [BHRgoal/CLAUDE.md](../BHRgoal/CLAUDE.md) - BHR algorithms to test

---

## Contribution Guidelines

### Writing Good Tests

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Describe what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Meaningful Assertions**: Use specific checks, not just "no crash"
5. **Edge Cases**: Test boundaries (empty, max, min, etc.)

### Example Contribution Workflow

```bash
# 1. Fork repository and create branch
git checkout -b add-algorithm-tests

# 2. Write test file
cd tests
vim test_foo.cpp

# 3. Update Makefile
# Add: OBJS += test_foo.o

# 4. Build and run
make
./runtests

# 5. Commit and submit PR
git add test_foo.cpp Makefile
git commit -m "Add FOO algorithm tests"
git push origin add-algorithm-tests
# Create GitHub Pull Request
```

### Test Review Criteria

When submitting tests, ensure:
- Tests compile without warnings
- All tests pass on clean build
- Tests are deterministic (no randomness)
- Tests execute quickly (< 1 second per test)
- Clear test names and descriptions
- Comments explaining complex test logic

---

## Future Work

### Test Infrastructure Improvements

1. **Continuous Integration**
   - GitHub Actions for automated testing
   - Test on multiple platforms (Linux, macOS)
   - Test with multiple compilers (g++, clang++)

2. **Coverage Reporting**
   - Add gcov/lcov for code coverage
   - Set coverage targets (e.g., 80%)
   - Identify untested code

3. **Test Organization**
   - Separate unit tests from integration tests
   - Add performance benchmarks
   - Add regression test suite

4. **Mocking**
   - Mock LEMON library for faster tests
   - Mock file I/O for trace parsing tests

5. **Documentation**
   - Add test writing guide
   - Document test coverage metrics
   - Create testing best practices guide
