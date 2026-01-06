#include <iostream>
#include <map>
#include <cassert>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <utility>
#include "parse_trace.h"
#include "../../../lib/trace/oracle_general_reader.h"

void parseTraceFile(std::vector<trEntry> & trace, std::string & path, uint64_t & byteSum) {
    OracleGeneralReader reader(path);
    std::vector<OracleGeneralReader::Record> records;

    if (!reader.readAll(records)) {
        std::cerr << "Error reading trace: " << reader.getError() << std::endl;
        return;
    }

    uint64_t time, id, size, reqc=0;
    std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t> lastSeen;

    for (const auto& rec : records) {
        id = rec.obj_id;
        size = rec.obj_size;
        time = rec.timestamp;

        const auto idsize = std::make_pair(id,size);
        if(size > 0 && lastSeen.count(idsize)>0) {
            trace[lastSeen[idsize]].hasNext = true;
            const uint64_t volume = (reqc-lastSeen[idsize]) * size;
            trace[lastSeen[idsize]].volume = volume;
        }
        trace.emplace_back(size);
        byteSum += size;
        lastSeen[idsize]=reqc++;
    }
}
