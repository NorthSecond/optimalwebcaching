#include "parse_trace.h"
#include "../../../lib/trace/oracle_general_reader.h"
#include <map>
#include <iostream>
#include <utility>

void parseTraceFile(std::vector<trEntry> & trace, std::string & path) {
    OracleGeneralReader reader(path);
    std::vector<OracleGeneralReader::Record> records;

    if (!reader.readAll(records)) {
        std::cerr << "Error reading trace: " << reader.getError() << std::endl;
        return;
    }

    std::map<std::pair<uint64_t, uint64_t>, size_t> lastSeen;
    uint64_t uniqc = 0;

    trace.clear();
    trace.reserve(records.size());

    for (const auto& rec : records) {
        uint64_t id = rec.obj_id;
        uint64_t size = rec.obj_size;

        auto key = std::make_pair(id, size);
        if (lastSeen.count(key) > 0) {
            trace[lastSeen[key]].hasNext = true;
        } else {
            uniqc++;
        }

        trace.push_back(trEntry(id, size));
        lastSeen[key] = trace.size() - 1;
    }
}
