#include "oracle_general_reader.h"
#include <cstring>
#include <cerrno>
#include <cstdio>

#ifdef USE_ZSTD
#include <zstd.h>  // Requires libzstd-dev
#endif

constexpr size_t RECORD_SIZE = 24;  // 4 + 8 + 4 + 8 bytes
constexpr size_t BUFFER_SIZE = 1024 * 1024;  // 1MB buffer

OracleGeneralReader::OracleGeneralReader(const std::string& path)
    : filepath_(path), record_count_(0) {
}

OracleGeneralReader::~OracleGeneralReader() = default;

bool OracleGeneralReader::readAll(std::vector<Record>& records) {
    records.clear();
    record_count_ = 0;

    FILE* file = fopen(filepath_.c_str(), "rb");
    if (!file) {
        error_msg_ = "Failed to open file: " + filepath_ + " (" + strerror(errno) + ")";
        return false;
    }

    // Detect compression by file extension
    bool is_compressed = filepath_.size() > 4 &&
                         filepath_.substr(filepath_.size() - 4) == ".zst";

    bool success = is_compressed ?
        readCompressed(file, records) :
        readUncompressed(file, records);

    fclose(file);
    return success;
}

bool OracleGeneralReader::readUncompressed(FILE* file, std::vector<Record>& records) {
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size < 0 || file_size % RECORD_SIZE != 0) {
        error_msg_ = "Invalid file size: " + std::to_string(file_size) +
                     " (not a multiple of " + std::to_string(RECORD_SIZE) + ")";
        return false;
    }

    size_t num_records = file_size / RECORD_SIZE;

    // Read entire file into buffer
    std::vector<char> buffer(file_size);
    size_t bytes_read = fread(buffer.data(), 1, file_size, file);
    if (bytes_read != static_cast<size_t>(file_size)) {
        error_msg_ = "Failed to read complete file";
        return false;
    }

    // Parse records
    records.resize(num_records);
    for (size_t i = 0; i < num_records; ++i) {
        const char* ptr = buffer.data() + i * RECORD_SIZE;
        if (!parseRecord(ptr, records[i])) {
            error_msg_ = "Failed to parse record at index " + std::to_string(i);
            return false;
        }
    }

    record_count_ = num_records;
    return true;
}

bool OracleGeneralReader::readCompressed(FILE* file, std::vector<Record>& records) {
#ifndef USE_ZSTD
    error_msg_ = "zstd compression support not enabled (compile with -DUSE_ZSTD)";
    fclose(file);
    return false;
#else
    // Get compressed file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate input buffer
    std::vector<char> compressed_data(file_size);
    size_t bytes_read = fread(compressed_data.data(), 1, file_size, file);
    if (bytes_read != static_cast<size_t>(file_size)) {
        error_msg_ = "Failed to read compressed file";
        return false;
    }

    // Create decompression context
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) {
        error_msg_ = "Failed to create zstd decompression context";
        return false;
    }

    // Estimate decompressed size (conservative estimate)
    size_t estimated_size = ZSTD_getFrameContentSize(compressed_data.data(), file_size);
    if (estimated_size == ZSTD_CONTENTSIZE_ERROR) {
        error_msg_ = "Invalid zstd frame";
        ZSTD_freeDCtx(dctx);
        return false;
    }
    if (estimated_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        estimated_size = file_size * 4;  // Conservative guess
    }

    // Allocate output buffer
    std::vector<char> decompressed_data(estimated_size);

    // Decompress
    size_t decompressed_size = ZSTD_decompressDCtx(
        dctx,
        decompressed_data.data(), estimated_size,
        compressed_data.data(), file_size
    );

    ZSTD_freeDCtx(dctx);

    if (ZSTD_isError(decompressed_size)) {
        error_msg_ = "Decompression failed: " + std::string(ZSTD_getErrorName(decompressed_size));
        return false;
    }

    // Verify decompressed size
    if (decompressed_size % RECORD_SIZE != 0) {
        error_msg_ = "Decompressed data size invalid: " +
                     std::to_string(decompressed_size) +
                     " (not a multiple of " + std::to_string(RECORD_SIZE) + ")";
        return false;
    }

    // Parse records
    size_t num_records = decompressed_size / RECORD_SIZE;
    records.resize(num_records);

    const char* ptr = decompressed_data.data();
    for (size_t i = 0; i < num_records; ++i, ptr += RECORD_SIZE) {
        if (!parseRecord(ptr, records[i])) {
            error_msg_ = "Failed to parse record at index " + std::to_string(i);
            return false;
        }
    }

    record_count_ = num_records;
    return true;
#endif
}

bool OracleGeneralReader::parseRecord(const char* buffer, Record& record) {
    // Parse binary format (little-endian assumed)
    std::memcpy(&record.timestamp, buffer, 4);
    std::memcpy(&record.obj_id, buffer + 4, 8);
    std::memcpy(&record.obj_size, buffer + 12, 4);
    std::memcpy(&record.next_access_vtime, buffer + 16, 8);

    // Validate
    if (record.obj_size == 0) {
        // Skip zero-size objects
        return false;
    }

    return true;
}
