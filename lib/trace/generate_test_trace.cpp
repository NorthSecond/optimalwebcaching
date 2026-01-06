#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <cstring>

struct OracleGeneralRecord {
    uint32_t timestamp;
    uint64_t obj_id;
    uint32_t obj_size;
    int64_t  next_access_vtime;
};

void write_binary_trace(const std::string& filename, const std::vector<OracleGeneralRecord>& records) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        exit(1);
    }

    for (const auto& rec : records) {
        // Write fields individually to avoid struct padding issues
        out.write(reinterpret_cast<const char*>(&rec.timestamp), 4);
        out.write(reinterpret_cast<const char*>(&rec.obj_id), 8);
        out.write(reinterpret_cast<const char*>(&rec.obj_size), 4);
        out.write(reinterpret_cast<const char*>(&rec.next_access_vtime), 8);
    }

    out.close();
    std::cout << "Wrote " << records.size() << " records to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_prefix>" << std::endl;
        return 1;
    }

    std::string prefix = argv[1];

    // Test trace 1: Simple sequence with repeats
    std::vector<OracleGeneralRecord> trace1 = {
        {1000, 1, 1024, 2000},   // t1: obj1, next at t2
        {2000, 2, 2048, 3000},   // t2: obj2, next at t3
        {3000, 1, 1024, -1},     // t3: obj1 again, no next
        {4000, 3, 512, 5000},    // t4: obj3, next at t5
        {5000, 2, 2048, -1},     // t5: obj2 again, no next
        {6000, 3, 512, -1},      // t6: obj3 again, no next
    };
    write_binary_trace(prefix + "_simple.dat", trace1);

    // Test trace 2: Size variation (same id, different size)
    std::vector<OracleGeneralRecord> trace2 = {
        {1000, 1, 1024, 2000},   // obj1 size 1024
        {2000, 1, 2048, 3000},   // obj1 size 2048 (different object!)
        {3000, 1, 1024, -1},     // obj1 size 1024 again
    };
    write_binary_trace(prefix + "_size_var.dat", trace2);

    // Test trace 3: Large cache pressure
    std::vector<OracleGeneralRecord> trace3;
    uint64_t obj_id = 1;
    for (uint32_t t = 1000; t <= 10000; t += 100) {
        trace3.push_back({t, obj_id, 1024 * 1024, (obj_id < 10) ? t + 1000 : -1});
        if (obj_id++ >= 10) obj_id = 1;
    }
    write_binary_trace(prefix + "_large.dat", trace3);

    // Test trace 4: Single object repeated
    std::vector<OracleGeneralRecord> trace4;
    for (uint32_t t = 1000; t <= 10000; t += 500) {
        trace4.push_back({t, 42, 4096, t + 500});
    }
    trace4.back().next_access_vtime = -1;  // Last one has no next
    write_binary_trace(prefix + "_single_obj.dat", trace4);

    std::cout << "\nAll test traces generated successfully!" << std::endl;
    std::cout << "To compress with zstd:" << std::endl;
    std::cout << "  zstd " << prefix << "_simple.dat -o " << prefix << "_simple.dat.zst" << std::endl;

    return 0;
}
