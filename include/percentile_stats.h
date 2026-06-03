// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Disk-based Index Construction and Utilities
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float tail_io_us = 0; // blocking IO wait time — tail end of IO flight when CPU is idle
    float cpu_us = 0;   // total time spent in CPU
    float selection_us = 0; // total time spent selecting nodes to expand
    
    // Detailed timing breakdown
    float init_us = 0;              // initialization time
    float nav_us = 0;               // navigation phase time
    float gap_io_wait_us = 0;       // pure I/O wait after cached processing
    float io_submit_us = 0;         // I/O submission time
    
    float io_flight_us = 0; // wall-clock IO in-flight time (submit → last completion, includes CPU overlap)
    float cached_full_data_during_io_us = 0;
    float uncached_full_data_during_io_us = 0;
    float uncached_full_data_outside_IO_us = 0;
    float deferred_full_vector_during_io_us = 0;
    float remain_full_vector_outside_io_us = 0;
    float io_completion_check_us = 0;
    float rest_cached_vectors_proc_time_during_io_us = 0;
    float cached_neighbors_out_io_us = 0;
    float node_proc_us = 0;

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops
    unsigned n_lsh_entry_points = 0;
    unsigned nnbr_explored = 0;
    unsigned n_mega_nodes_processed = 0; // # mega nodes processed (for avg nodes per hop)
    unsigned n_non_hub_requested = 0;    // # non-hub vectors requested for disk I/O
    unsigned n_useful_ios = 0;           // # IO pages whose vectors appear in the final retset

    // Cache-aware expansion statistics (LAANN only)
    unsigned n_window_cached = 0;    // # of cached expansions within candidate window
    unsigned n_window_uncached = 0;  // # of uncached expansions within candidate window (persistent + fresh uncached)
    unsigned n_outside_cached = 0;   // # of cached expansions outside candidate window
    unsigned n_outside_uncached = 0; // # of uncached expansions outside candidate window
    unsigned n_persistent_uncached = 0; // # of persistent uncached expansions (Priority 1)

    // CPU overhead breakdown (LAANN only)
    float memcpy_us = 0;           // Time spent in memcpy operations
    float hash_lookup_us = 0;      // Time spent in hash map lookups
    float classification_us = 0;   // Time spent classifying cached vs uncached
    float buffer_mgmt_us = 0;      // Time spent in buffer management (push_back, etc.)
    unsigned n_classification_lookups = 0;  // # of hash lookups for classification
    unsigned n_retrieval_lookups = 0;       // # of hash lookups for retrieval

    // NOTE: cpu_us field is reused to track actual CPU processing time
    // It accumulates time from cached_processing + uncached_processing loops
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
