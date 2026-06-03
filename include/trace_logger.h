// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>

namespace diskann {

class TraceLogger {
private:
    std::ofstream trace_file;
    std::mutex write_mutex;
    std::atomic<uint64_t> access_count;
    bool enabled;

public:
    TraceLogger() : access_count(0), enabled(false) {}

    ~TraceLogger() {
        close();
    }

    bool open(const std::string& filename) {
        if (filename.empty()) return false;

        std::lock_guard<std::mutex> lock(write_mutex);
        trace_file.open(filename, std::ios::out);
        if (trace_file.is_open()) {
            // Write CSV header
            trace_file << "timestamp,obj_id,obj_size\n";
            trace_file.flush();
            enabled = true;
            return true;
        }
        return false;
    }

    void close() {
        std::lock_guard<std::mutex> lock(write_mutex);
        if (trace_file.is_open()) {
            trace_file.close();
        }
        enabled = false;
    }

    void log_access(uint32_t mega_node_id, uint64_t size_bytes) {
        if (!enabled || !trace_file.is_open()) return;

        uint64_t timestamp = access_count.fetch_add(1) + 1;

        std::lock_guard<std::mutex> lock(write_mutex);
        trace_file << timestamp << ","
                   << mega_node_id << ","
                   << size_bytes << "\n";
    }

    void flush() {
        std::lock_guard<std::mutex> lock(write_mutex);
        if (trace_file.is_open()) {
            trace_file.flush();
        }
    }

    bool is_enabled() const { return enabled; }
    uint64_t get_access_count() const { return access_count.load(); }
};

// Global trace logger instance (for one-time use without changing signatures)
extern std::unique_ptr<TraceLogger> g_trace_logger;

// Helper functions
inline void init_global_trace_logger(const std::string& filename) {
    g_trace_logger = std::make_unique<TraceLogger>();
    if (g_trace_logger->open(filename)) {
        diskann::cout << "[TRACE] Trace logging enabled: " << filename << std::endl;
    }
}

inline void close_global_trace_logger() {
    if (g_trace_logger) {
        g_trace_logger->flush();
        g_trace_logger->close();
        diskann::cout << "[TRACE] Trace logging completed. Total accesses: "
                      << g_trace_logger->get_access_count() << std::endl;
    }
}

inline void log_mega_node_access(uint32_t mega_node_id, uint64_t size_bytes) {
    if (g_trace_logger && g_trace_logger->is_enabled()) {
        g_trace_logger->log_access(mega_node_id, size_bytes);
    }
}

} // namespace diskann
