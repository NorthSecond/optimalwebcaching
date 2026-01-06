#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <functional>

// Trace entry structure (compatible with all algorithms)
struct trEntry {
    uint64_t id;
    uint64_t size;
    size_t nextSeen;
    bool hasNext;
    bool hit;

    trEntry(uint64_t nid, uint64_t nsize)
        : id(nid),
          size(nsize),
          nextSeen(0),
          hasNext(false),
          hit(false)
    {}
};

// Extended trace entry with time and arcId (for algorithms that need it)
struct trEntryExtended : public trEntry {
    uint64_t time;
    int arcId;

    trEntryExtended(uint64_t nid, uint64_t nsize, uint64_t ntime)
        : trEntry(nid, nsize),
          time(ntime),
          arcId(-1)
    {}
};

// Hash combine for std::pair
template<class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
    template<typename S, typename T>
    struct hash<pair<S, T>> {
        inline size_t operator()(const pair<S, T>& v) const {
            size_t seed = 0;
            ::hash_combine(seed, v.first);
            ::hash_combine(seed, v.second);
            return seed;
        }
    };
}

/**
 * Parse OracleGeneral format trace (binary, optionally zstd compressed)
 *
 * @param trace Output vector of trace entries
 * @param path Path to trace file (.zst extension indicates compression)
 * @return Number of unique objects, or 0 on error
 */
uint64_t parseOracleGeneralTrace(std::vector<trEntry>& trace, const std::string& path);

/**
 * Parse OracleGeneral format trace with extended information (time, arcId)
 *
 * @param trace Output vector of extended trace entries
 * @param path Path to trace file
 * @return Number of unique objects, or 0 on error
 */
uint64_t parseOracleGeneralTraceExtended(std::vector<trEntryExtended>& trace, const std::string& path);

/**
 * Legacy function for backward compatibility (now uses OracleGeneral format)
 */
uint64_t parseTraceFile(std::vector<trEntry>& trace, std::string& path);
