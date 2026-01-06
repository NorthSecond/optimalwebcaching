#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

/**
 * OracleGeneral Trace Reader with zstd compression support
 *
 * Binary format (24 bytes per record):
 * struct {
 *     uint32_t timestamp;        // 4 bytes
 *     uint64_t obj_id;           // 8 bytes
 *     uint32_t obj_size;         // 4 bytes
 *     int64_t  next_access_vtime;// 8 bytes (-1 if no next access)
 * }
 */

class OracleGeneralReader {
public:
    struct Record {
        uint32_t timestamp;
        uint64_t obj_id;
        uint32_t obj_size;
        int64_t  next_access_vtime;
    };

    explicit OracleGeneralReader(const std::string& path);
    ~OracleGeneralReader();

    // Read all records into vector
    bool readAll(std::vector<Record>& records);

    // Get error message if operation failed
    const std::string& getError() const { return error_msg_; }

    // Get total number of records
    size_t getRecordCount() const { return record_count_; }

private:
    std::string filepath_;
    std::string error_msg_;
    size_t record_count_;

    // Read uncompressed data
    bool readUncompressed(FILE* file, std::vector<Record>& records);

    // Read zstd compressed data
    bool readCompressed(FILE* file, std::vector<Record>& records);

    // Parse binary record
    bool parseRecord(const char* buffer, Record& record);
};
