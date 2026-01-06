TARGET = foo
OBJS += lib/parse_trace.o
OBJS += lib/solve_mcf.o
OBJS += ../../lib/trace/oracle_general_reader.o
OBJS += foo.o

CXX = g++ #clang++ #OSX
CXXFLAGS += -std=c++11 #-stdlib=libc++ #non-linux
CXXFLAGS += -MMD -MP # dependency tracking flag
CXXFLAGS += -I.. -I ../../lib
CXXFLAGS += -O3 -ffast-math -march=native # aggressive optimization settings
CXXFLAGS += -Wall
# Disable maybe-uninitialized warnings from LEMON library
CXXFLAGS += -Wno-maybe-uninitialized
# Uncomment the following to enable stricter error checking (may fail with LEMON)
# CXXFLAGS += -Werror
# Uncomment to enable zstd support
# CXXFLAGS += -DUSE_ZSTD
# LDFLAGS += -lzstd

all:		$(TARGET)

debug: CXXFLAGS += -ggdb  -D_GLIBCXX_DEBUG # debug flags
debug: $(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	-rm $(TARGET) $(OBJS) $(DEPS)
