#include "parse_trace.h"
#include "oracle_general_reader.h"
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

uint64_t parseOracleGeneralTrace(std::vector<trEntry>& trace, const std::string& path) {
    OracleGeneralReader reader(path);
    std::vector<OracleGeneralReader::Record> records;

    if (!reader.readAll(records)) {
        std::cerr << "Error reading trace: " << reader.getError() << std::endl;
        return 0;
    }

    // Map to track last seen position for each (id, size) pair
    std::map<std::pair<uint64_t, uint64_t>, size_t> lastSeen;
    uint64_t uniqc = 0;

    trace.clear();
    trace.reserve(records.size());

    for (size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        uint64_t id = rec.obj_id;
        uint64_t size = rec.obj_size;

        // Check if this (id, size) was seen before
        auto key = std::make_pair(id, size);
        if (lastSeen.count(key) > 0) {
            // Mark previous request as having a next request
            trace[lastSeen[key]].hasNext = true;
        } else {
            uniqc++;
        }

        // Add new trace entry
        trace.push_back(trEntry(id, size));
        lastSeen[key] = trace.size() - 1;
    }

    return uniqc;
}

uint64_t parseOracleGeneralTraceExtended(std::vector<trEntryExtended>& trace, const std::string& path) {
    OracleGeneralReader reader(path);
    std::vector<OracleGeneralReader::Record> records;

    if (!reader.readAll(records)) {
        std::cerr << "Error reading trace: " << reader.getError() << std::endl;
        return 0;
    }

    // Map to track last seen position for each (id, size) pair
    std::map<std::pair<uint64_t, uint64_t>, size_t> lastSeen;
    uint64_t uniqc = 0;

    trace.clear();
    trace.reserve(records.size());

    for (size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        uint64_t id = rec.obj_id;
        uint64_t size = rec.obj_size;
        uint64_t time = rec.timestamp;

        // Check if this (id, size) was seen before
        auto key = std::make_pair(id, size);
        if (lastSeen.count(key) > 0) {
            // Mark previous request as having a next request
            trace[lastSeen[key]].hasNext = true;
        } else {
            uniqc++;
        }

        // Add new trace entry with time
        trace.emplace_back(id, size, time);
        lastSeen[key] = trace.size() - 1;
    }

    return uniqc;
}

// Legacy function for backward compatibility
uint64_t parseTraceFile(std::vector<trEntry>& trace, std::string& path) {
    return parseOracleGeneralTrace(trace, path);
}
