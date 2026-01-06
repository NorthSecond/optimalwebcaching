# Makefile for trace library

CXX = g++
CXXFLAGS += -std=c++11
CXXFLAGS += -Wall -Wextra

# Uncomment to enable zstd support
# CXXFLAGS += -DUSE_ZSTD
# LDFLAGS += -lzstd

OBJS = oracle_general_reader.o parse_trace.o
TEST_GEN = generate_test_trace

all: $(OBJS)

test-gen: generate_test_trace.cpp
	$(CXX) $(CXXFLAGS) -o $(TEST_GEN) $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	-rm $(OBJS) $(TEST_GEN)

.PHONY: all test-gen clean
