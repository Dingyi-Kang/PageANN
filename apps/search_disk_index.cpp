// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Unified DiskANN Search Tool: PageANN and LAANN
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

/**
 * @file search_disk_index.cpp
 * @brief Unified CLI tool to search both PageANN and LAANN disk indexes.
 *
 * This tool provides a unified interface for two disk-based ANN search implementations:
 * - PageANN (default): Page-based static caching with optimized graph layout
 * - LAANN (--use_laann flag): Batch-interleaved processing with dynamic caching
 *
 * Both implementations support:
 * - Product Quantization (PQ) for fast distance approximation
 * - Hash-based routing for entry point selection (optional)
 * - Parallel search with configurable thread count
 * - Multiple search list sizes (L values) for recall/latency trade-offs
 *
 * @usage ./search_disk_index --data_type <float|int8|uint8> --dist_fn <l2|mips|cosine>
 *        --index_path_prefix <index_prefix> --query_file <queries.bin>
 *        -K <recall_at> -L <search_list_sizes>
 *        [--num_pages_to_cache <pages>] [--beamwidth <W>] [--num_threads <T>]
 * *        [--use_laann]
 */
#include "common_includes.h"
#include <boost/program_options.hpp>


#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "pq_flash_index_laann.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "trace_logger.h"
#include <fstream>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

/**
 * @brief Execute approximate nearest neighbor search on page-based disk index.
 *
 * This function performs the complete search workflow:
 * 1. Load index and PQ data
 * 2. Generate cache list from sample queries (optional)
 * 3. Load frequently-accessed pages into memory cache
 * 4. Execute parallel search for all query vectors
 * 5. Calculate recall metrics and report performance statistics
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @tparam LabelT Label type (default: uint32_t)
 * @param metric Distance metric (L2, INNER_PRODUCT, or COSINE)
 * @param index_path_prefix Path prefix for page index files
 * @param pq_path_prefix Path prefix for PQ data files
 * @param query_file Path to query vectors file
 * @param gt_file Path to ground truth file (for recall calculation)
 * @param num_threads Number of parallel threads for search
 * @param recall_at Number of nearest neighbors to retrieve (K)
 * @param beamwidth Beam width for beam search (0 = auto-optimize)
 * @param num_pages_to_cache Number of frequently-visited pages to cache
 * @param search_io_limit Maximum I/O operations per query
 * @param Lvec List of search list sizes (L) to evaluate
 * @param fail_if_recall_below Minimum acceptable recall (exit with error if not met)
 * @param query_filters Filter labels for filtered search (not currently used)
 * @param use_reorder_data Use full precision data for reranking
 * @param nav_L Navigation graph search depth (0 = disabled, >0 = enabled)
 * @return 0 on success, -1 on failure or insufficient recall
 */
template <typename T, typename LabelT = uint32_t>
int search_pageann_disk_index(diskann::Metric &metric, const std::string &index_path_prefix, const std::string &pq_path_prefix,
                      const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_pages_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false, const uint32_t nav_L = 0,
                      const float cache_ratio = 0.0f)
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    // ===== STEP 1: Load query vectors and ground truth data =====
    std::string warmup_query_file = "";  // Warmup queries not currently used

    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;  // Filtered search not currently supported

    // Load ground truth for recall calculation (if provided)
    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    // ===== STEP 2: Initialize PQFlashIndex and load index from disk =====
    // Create platform-specific aligned file reader
    std::shared_ptr<AlignedFileReader> dataReader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    dataReader.reset(new WindowsAlignedFileReader());
#else
    dataReader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    dataReader.reset(new LinuxAlignedFileReader());
#endif

    // Create PQFlashIndex instance with file reader and distance metric
    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(dataReader, metric));

    // Load page-based index, PQ data, and optionally hash routing structures
    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix);

    // Load navigation graph if requested
    if (nav_L > 0) {
        _pFlashIndex->load_nav_graph(index_path_prefix.c_str());
    }

    if (res != 0)
    {
        return res;
    }

    // ===== STEP 3: Generate cache list and load frequently-accessed pages =====
    uint32_t actual_pages_to_cache = num_pages_to_cache;
    if (cache_ratio > 0.0f && actual_pages_to_cache == 0) {
        actual_pages_to_cache = static_cast<uint32_t>(_pFlashIndex->get_num_pages() * cache_ratio);
        diskann::cout << "Cache ratio " << cache_ratio << " -> num_pages_to_cache = " << actual_pages_to_cache << std::endl;
    }
    std::vector<uint32_t> page_list;
    diskann::cout << "Caching " << actual_pages_to_cache << " most frequently visited pages based on sample data." << std::endl;

    // Use sample queries to identify most frequently visited pages
    // This enables intelligent caching by profiling actual search patterns
    std::string pageANN_warmup_query_file = index_path_prefix + "_sample_data.bin";
    if (actual_pages_to_cache > 0)
    {
        std::string cache_order_file = index_path_prefix + "_cache_page_order.bin";
        if (file_exists(cache_order_file))
        {
            std::ifstream ifs(cache_order_file, std::ios::binary);
            uint64_t total_pages;
            ifs.read((char *)&total_pages, sizeof(uint64_t));
            uint64_t pages_to_load = std::min((uint64_t)actual_pages_to_cache, total_pages);
            page_list.resize(pages_to_load);
            ifs.read((char *)page_list.data(), pages_to_load * sizeof(uint32_t));
            ifs.close();
            diskann::cout << "Loaded " << pages_to_load << " cached page IDs from "
                          << cache_order_file << " (total saved: " << total_pages << ")"
                          << std::endl;
        }
        else
        {
            const uint64_t num_pages_to_save = _pFlashIndex->get_num_pages();
            _pFlashIndex->generate_cache_list_from_sample_queries(
                pageANN_warmup_query_file, Lvec[0], beamwidth, num_pages_to_save, num_threads, page_list, nav_L);

            std::ofstream ofs(cache_order_file, std::ios::binary);
            uint64_t total_pages = page_list.size();
            ofs.write((char *)&total_pages, sizeof(uint64_t));
            ofs.write((char *)page_list.data(), total_pages * sizeof(uint32_t));
            ofs.close();
            diskann::cout << "Saved " << total_pages << " cached page IDs to "
                          << cache_order_file << std::endl;

            page_list.resize(actual_pages_to_cache);
        }
    }

    // Load the identified pages into memory cache (neighbor lists and optionally vector data)
    _pFlashIndex->load_cache_list(page_list);

    // Free memory used by page_list after caching is complete
    page_list.clear();
    page_list.shrink_to_fit();

    // ===== STEP 4: Optionally perform warmup queries =====
    omp_set_max_active_levels(2);  // Allow 2 levels of nested parallelism
    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    // Warmup helps prime caches and stabilize performance measurements (currently disabled by default)
    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->page_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    // ===== STEP 5: Initialize performance reporting =====
    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    // Determine recall levels to calculate based on recall_at
    std::vector<uint32_t> recall_levels;
    std::vector<std::string> recall_labels;

    // Always include recall_at as the primary level
    recall_levels.push_back(recall_at);
    recall_labels.push_back("Recall@" + std::to_string(recall_at));

    // Add smaller standard levels that are less than recall_at
    std::vector<uint32_t> standard_levels = {10, 5, 2, 1};
    for (auto level : standard_levels)
    {
        if (level < recall_at)
        {
            recall_levels.push_back(level);
            recall_labels.push_back("Recall@" + std::to_string(level));
        }
    }

    // Print cache configuration info for original PageANN
    uint64_t total_pages = _pFlashIndex->get_num_pages();
    uint64_t cached_pages = _pFlashIndex->get_num_cached_pages();
    double cache_pct = (total_pages > 0) ? (100.0 * cached_pages / total_pages) : 0.0;

    diskann::cout << "\n" << std::string(120, '=') << std::endl;
    diskann::cout << "SEARCH PERFORMANCE RESULTS" << std::endl;
    diskann::cout << "Index: " << total_pages << " pages, "
                  << cached_pages << " cached ("
                  << std::fixed << std::setprecision(1) << cache_pct << "%), "
                  << "T=" << num_threads << std::endl;
    diskann::cout << std::string(120, '=') << std::endl;

    diskann::cout << std::left
                  << std::setw(6) << "L"
                  << std::setw(4) << "B"
                  << std::setw(10) << "QPS"
                  << std::setw(15) << "Latency (us)"
                  << std::setw(12) << "IO (us)"
                  << std::setw(12) << "CPU (us)"
                  << std::setw(12) << "Lat-IO(us)"
                  << std::setw(10) << "Mean IOs"
                  << std::setw(12) << "Pool Pages"
                  << std::setw(8) << "Hops"
                  << std::setw(10) << "Nodes/Hop"
                  << std::setw(12) << "CacheHit%";
    if (calc_recall_flag)
    {
        for (const auto& label : recall_labels)
        {
            diskann::cout << std::setw(12) << label;
        }
    }
    diskann::cout << std::endl;
    diskann::cout << std::string(120, '-') << std::endl;

    // ===== STEP 6: Execute search for each L value and measure performance =====
    // Buffers to store search results for all L values
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;
    double best_recall = 0.0;

    // Test each search list size (L) parameter
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        // Auto-tune beamwidth if not specified (beamwidth=0 triggers optimization)
        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;
        
        // Allocate result buffers and statistics tracking for this L value
        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        auto stats = new diskann::QueryStats[query_num];
        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);

        // Execute parallel search across all queries
        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                // Platform-specific search: page_search (Windows) or linux_page_search (Linux)
#ifdef _WINDOWS
                _pFlashIndex->page_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i, nav_L);
#else
                _pFlashIndex->linux_page_search(query + (i * query_aligned_dim), recall_at, L,
                                               query_result_ids_64.data() + (i * recall_at),
                                               query_result_dists[test_id].data() + (i * recall_at),
                                               optimized_beamwidth, use_reorder_data, stats + i, nav_L);
#endif
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        // Convert result IDs from 64-bit to 32-bit format
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(), query_num, recall_at);

        // Compute performance statistics across all queries
        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto mean_avg_nodes_per_hop = diskann::get_mean_stats<float>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_hops > 0 ? (float)stats.n_mega_nodes_processed / stats.n_hops : 0.0f; });
        auto mean_tail_io_us = diskann::get_mean_stats<float>(stats, query_num, [](const diskann::QueryStats &stats) { return stats.tail_io_us; });

        auto io_999 = diskann::get_percentile_stats<float>(
                    stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.tail_io_us; });

        auto mean_cpu_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_cache_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cache_hits; });
        
        float cache_hit_ratio = 0.0f;
        if (mean_cache_hits + mean_ios > 0) {
            cache_hit_ratio = (mean_cache_hits / (mean_cache_hits + mean_ios)) * 100.0f;
        }

        auto mean_lsh_entry_points = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_lsh_entry_points; });

        auto mean_nnbr_explored = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.nnbr_explored; });

        // Calculate recall metrics at different K values if ground truth is available
        std::vector<double> recall_values;
        if (calc_recall_flag)
        {
            for (auto level : recall_levels)
            {
                double recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                          query_result_ids[test_id].data(), recall_at, level);
                recall_values.push_back(recall);
            }
            // Track best recall (using the primary recall_at level)
            best_recall = std::max(recall_values[0], best_recall);
        }

        // Output performance results for this L value
        diskann::cout << std::left
                      << std::setw(6) << L
                      << std::setw(4) << optimized_beamwidth
                      << std::setw(10) << std::fixed << std::setprecision(1) << qps
                      << std::setw(15) << std::fixed << std::setprecision(2) << mean_latency
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_tail_io_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_cpu_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << (mean_latency - mean_tail_io_us)
                      << std::setw(10) << std::fixed << std::setprecision(1) << mean_ios
                      << std::setw(8) << std::fixed << std::setprecision(1) << mean_hops
                      << std::setw(10) << std::fixed << std::setprecision(1) << mean_avg_nodes_per_hop
                      << std::setw(12) << std::fixed << std::setprecision(2) << cache_hit_ratio;
        if (calc_recall_flag)
        {
            for (auto recall_val : recall_values)
            {
                diskann::cout << std::setw(12) << std::fixed << std::setprecision(4) << recall_val;
            }
        }
        diskann::cout << std::endl;

        delete[] stats;
    }

    diskann::cout << "Done searching. Not save results " << std::endl;

    // Clean up allocated memory
    diskann::aligned_free(query);
    if (gt_ids != nullptr) 
        delete[] gt_ids;
    if (gt_dists != nullptr) 
        delete[] gt_dists;
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

/**
 * @brief LAANN disk-based search implementation using beam search
 *
 * This function performs graph search on LAANN disk indexes using laann_search.
 * Uses beam width parameters with candidate window concept:
 * - cache_beam_width: for cache priority mode (processes cached pages)
 * - normal_search_beam_width: for normal disk I/O mode within candidate window
 * - out_window_beam_width: target beam width when going outside candidate window
 *
 * @param metric Distance metric (L2, COSINE, or INNER_PRODUCT)
 * @param index_path_prefix Path prefix for index files
 * @param pq_path_prefix Path prefix for PQ compressed data
 * @param query_file Path to query vectors file
 * @param gt_file Path to ground truth file (or "null" to skip recall calculation)
 * @param num_threads Number of parallel search threads
 * @param recall_at K value for recall@K calculation
 * @param search_beam_width Beam width used for all search modes (look-ahead and normal)
 * @param use_look_ahead_search Enable look-ahead search mode (collect cached pages first, skip disk until persistence check triggers normal mode)
 * @param num_pages_to_cache Number of frequently-accessed pages to cache in memory
 * @param search_io_limit Maximum number of IOs per search
 * @param Lvec List of L values (search list sizes) to test
 * @param fail_if_recall_below Minimum recall threshold (returns -1 if not met)
 * @param query_filters Vector of filter labels (currently not supported)
 * @param use_reorder_data Whether to use full-precision reordering data
 * @return 0 on success, -1 on failure or insufficient recall
 */
template <typename T, typename LabelT = uint32_t>
int search_laann_disk_index(diskann::Metric &metric, const std::string &index_path_prefix, const std::string &pq_path_prefix,
                      const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at,
                      const uint32_t search_beam_width,
                      const bool use_look_ahead_search, const bool use_pipeline,
                      const uint32_t persistence_window_width,
                      const uint32_t num_pages_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                      const uint32_t nav_L = 14, const float cache_ratio = 0.0f,
                      const float retset_capacity_ratio = 2.0f,
                      const uint32_t stable_rank_threshold = 10,
                      const float beamwidth_spike_ratio = 0.5f,
                      const float beam_decay_ratio = 0.9f,
                      const std::string &cache_order_file = "")
{
    diskann::cout << "Search parameters: #threads: " << num_threads;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    // ===== Initialize async I/O BEFORE load (so thread setup has the file path) =====
    std::shared_ptr<AlignedFileReader> reader = nullptr;
    reader.reset(new LinuxAlignedFileReader());

    std::unique_ptr<diskann::PQFlashIndexLAANN<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndexLAANN<T, LabelT>(reader, metric));

    int load_result = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix.c_str(), search_beam_width);

    if (load_result != 0)
    {
        diskann::cerr << "Failed to load index." << std::endl;
        return -1;
    }

    // ===== Load navigation neighbors (only if nav_L > 0) =====
    if (nav_L > 0)
    {
        _pFlashIndex->load_nav_neighbors(index_path_prefix.c_str());
    }

    // ===== Auto-generate order file if needed =====
    // If an order file path is given but the file doesn't exist yet, profile with sample queries
    // and write the sorted page-frequency file — same pattern as PageANN's cache list generation.
    if (!cache_order_file.empty() && !file_exists(cache_order_file)) {
        diskann::cout << "Order file not found. Generating via profiling -> " << cache_order_file << std::endl;
        std::string sample_bin = index_path_prefix + "_sample_data.bin";
        uint64_t profile_l = Lvec[0];
        _pFlashIndex->profile_page_frequency(sample_bin, profile_l, search_beam_width,
                                             num_threads, cache_order_file, nav_L);
    }

    // ===== Load pages into cache =====
    uint32_t actual_pages_to_cache = num_pages_to_cache;
    if (cache_ratio > 0.0f) {
        uint64_t total_pages = _pFlashIndex->get_num_mega_nodes();
        actual_pages_to_cache = static_cast<uint32_t>(total_pages * cache_ratio);
        diskann::cout << "Cache ratio " << cache_ratio << " -> caching " << actual_pages_to_cache
                      << " of " << total_pages << " pages" << std::endl;
    }
    if (actual_pages_to_cache > 0) {
        if (!cache_order_file.empty()) {
            // Non-reordered index: load pages by profile order file (non-contiguous disk reads)
            diskann::cout << "Using order-file cache: " << cache_order_file << std::endl;
            _pFlashIndex->load_order_file_page_cache(cache_order_file, actual_pages_to_cache);
        } else {
            // Reordered index: pages 0..N-1 are already the most frequently visited
            _pFlashIndex->load_sequential_page_cache(actual_pages_to_cache);
        }
    }

    omp_set_num_threads(num_threads);

    // ===== STEP 1: Load query vectors and ground truth data =====
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // Load ground truth for recall calculation (if provided)
    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error: Mismatch in number of queries. QueryFile: " << query_num
                         << ", TruthFile: " << gt_num << std::endl;
            return -1;
        }
        calc_recall_flag = true;
    }

    // ===== STEP 2: Initialize performance reporting =====
    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    // Determine recall levels to calculate based on recall_at
    std::vector<uint32_t> recall_levels;
    std::vector<std::string> recall_labels;

    recall_levels.push_back(recall_at);
    recall_labels.push_back("Recall@" + std::to_string(recall_at));

    std::vector<uint32_t> standard_levels = {10, 5, 2, 1};
    for (auto level : standard_levels)
    {
        if (level < recall_at)
        {
            recall_levels.push_back(level);
            recall_labels.push_back("Recall@" + std::to_string(level));
        }
    }

    // Print cache configuration info
    uint64_t total_mega_nodes = _pFlashIndex->get_num_mega_nodes();
    uint64_t cached_mega_nodes = _pFlashIndex->get_num_cached_mega_nodes();
    double cache_ratio_mega = (total_mega_nodes > 0) ? (100.0 * cached_mega_nodes / total_mega_nodes) : 0.0;

    diskann::cout << "\n" << std::string(140, '=') << std::endl;
    diskann::cout << "LAANN BEAM SEARCH RESULTS" << std::endl;
    diskann::cout << "Index: " << total_mega_nodes << " mega nodes, "
                  << cached_mega_nodes << " cached ("
                  << std::fixed << std::setprecision(1) << cache_ratio_mega << "%), "
                  << "T=" << num_threads << std::endl;
    diskann::cout << "BeamWidth=" << search_beam_width
                  << ", LookAhead=" << std::boolalpha << use_look_ahead_search
                  << ", PersistWindow=" << persistence_window_width
                  << ", Pipeline=" << std::boolalpha << use_pipeline
                  << ", nav_L=" << nav_L
                  << ", beamwidth_spike_ratio=" << beamwidth_spike_ratio
                  << ", beam_decay_ratio=" << beam_decay_ratio << std::endl;
    diskann::cout << std::string(140, '=') << std::endl;

    diskann::cout << std::left
                  << std::setw(6) << "L"
                  << std::setw(10) << "QPS"
                  << std::setw(12) << "Lat(us)"
                  << std::setw(10) << "IO(us)"
                  << std::setw(10) << "CPU(us)"
                  << std::setw(8) << "IOs"
                  << std::setw(7) << "Hops"
                  << std::setw(10) << "CacheHit%";
    if (calc_recall_flag)
    {
        for (const auto& label : recall_labels)
        {
            diskann::cout << std::setw(12) << label;
        }
    }
    diskann::cout << std::endl;
    diskann::cout << std::string(140, '-') << std::endl;

    // ===== STEP 3: Execute search for each L value =====
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        auto stats = new diskann::QueryStats[query_num];
        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++) {
            _pFlashIndex->laann_search(query + (i * query_aligned_dim), recall_at, L,
                                            query_result_ids_64.data() + (i * recall_at),
                                            query_result_dists[test_id].data() + (i * recall_at),
                                            search_beam_width,
                                            use_look_ahead_search, use_pipeline,
                                            persistence_window_width,
                                            nav_L, use_reorder_data, retset_capacity_ratio, stable_rank_threshold, beamwidth_spike_ratio, beam_decay_ratio, stats + i);
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        // Convert result IDs from 64-bit to 32-bit format
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(), query_num, recall_at);

        // Compute performance statistics
        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_tail_io_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.tail_io_us; });

        auto mean_cpu_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto mean_cache_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cache_hits; });

        // Calculate cache hit ratio
        float cache_hit_ratio = 0.0f;
        if (mean_cache_hits + mean_ios > 0) {
            cache_hit_ratio = (mean_cache_hits / (mean_cache_hits + mean_ios)) * 100.0f;
        }

        // Print results
        diskann::cout << std::left
                      << std::setw(6) << L
                      << std::setw(10) << std::fixed << std::setprecision(1) << qps
                      << std::setw(12) << std::fixed << std::setprecision(1) << mean_latency
                      << std::setw(10) << std::fixed << std::setprecision(1) << mean_tail_io_us
                      << std::setw(10) << std::fixed << std::setprecision(1) << mean_cpu_us
                      << std::setw(8) << std::fixed << std::setprecision(1) << mean_ios
                      << std::setw(7) << std::fixed << std::setprecision(1) << mean_hops
                      << std::setw(10) << std::fixed << std::setprecision(1) << cache_hit_ratio;

        // Calculate and print recall
        if (calc_recall_flag)
        {
            for (size_t j = 0; j < recall_levels.size(); j++)
            {
                auto level = recall_levels[j];
                double recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                                         query_result_ids[test_id].data(), recall_at, level);
                diskann::cout << std::setw(12) << std::fixed << std::setprecision(4) << recall;
                if (j == 0)
                    best_recall = std::max(best_recall, recall);
            }
        }

        diskann::cout << std::endl;

        // Print detailed timing breakdown
        auto mean_init_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.init_us; });
        auto mean_nav_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.nav_us; });
        auto mean_io_submit_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_submit_us; });
        auto mean_cached_proc_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.cached_full_data_during_io_us; });
        auto mean_uncached_proc_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.uncached_full_data_during_io_us; });
        auto mean_io_flight_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_flight_us; });
        float mean_cpu_overlap_us = mean_io_flight_us - mean_tail_io_us;
        float mean_total_cpu_us = mean_latency - mean_io_flight_us + mean_cpu_overlap_us;
        auto mean_cmps = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cmps; });
        auto mean_mega_nodes_processed = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_mega_nodes_processed; });

        diskann::cout << "  Timing breakdown (us): Init=" << std::fixed << std::setprecision(1) << mean_init_us
                      << ", Nav=" << mean_nav_us
                      << ", IO_submit_time=" << mean_io_submit_us
                      << ", Tail_IO=" << mean_tail_io_us
                      << ", Cached_proc=" << mean_cached_proc_us
                      << ", Disk_proc=" << mean_uncached_proc_us << std::endl;
        diskann::cout << "  CPU/IO summary (us): Total_IO_flight=" << std::fixed << std::setprecision(1) << mean_io_flight_us
                      << ", Tail_IO=" << mean_tail_io_us
                      << ", CPU_overlap_with_IO=" << mean_cpu_overlap_us
                      << ", Total_CPU=" << mean_total_cpu_us << std::endl;
        auto mean_non_hub_requested = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_non_hub_requested; });
        auto mean_useful_ios = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_useful_ios; });
        float useful_io_ratio = (mean_ios > 0) ? (mean_useful_ios / mean_ios) * 100.0f : 0.0f;

        diskann::cout << "  Stats: Cmps=" << std::fixed << std::setprecision(1) << mean_cmps
                      << ", MegaNodes_proc=" << mean_mega_nodes_processed
                      << ", NonHub_requested=" << mean_non_hub_requested
                      << ", Avg_nodes/hop=" << (mean_hops > 0 ? mean_mega_nodes_processed / mean_hops : 0.0)
                      << ", UsefulIO%=" << std::setprecision(1) << useful_io_ratio << std::endl;

        delete[] stats;
    }

    diskann::cout << std::string(140, '=') << std::endl;

    diskann::aligned_free(query);
    if (calc_recall_flag)
    {
        delete[] gt_ids;
        delete[] gt_dists;
    }

    return best_recall >= fail_if_recall_below ? 0 : -1;
}


int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, query_file, gt_file, pq_path_prefix, filter_label,
        label_type, query_filters_file, cache_order_file;
    uint32_t num_threads, K, W, num_pages_to_cache, nav_L, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;
    float cache_ratio = 0.0f;
    bool use_laann = false;
    bool use_look_ahead_search = true;
    bool use_pipeline = true;
    uint32_t persistence_window_width = 0;
    float retset_capacity_ratio = 2.0f;
    uint32_t stable_rank_threshold = 10;
    float beamwidth_spike_ratio = 0.5f;
    float beam_decay_ratio = 0.9f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters 
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("pq_path_prefix", po::value<std::string>(&pq_path_prefix)->default_value(std::string("")),
                                       "Path for PQ data");
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_pages_to_cache", po::value<uint32_t>(&num_pages_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()("cache_ratio", po::value<float>(&cache_ratio)->default_value(0.0f),
                                       "Ratio of pages to cache [0.0-1.0]. Overrides num_pages_to_cache if > 0.");
        optional_configs.add_options()("search_io_limit", po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
                                        "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("nav_L", po::value<uint32_t>(&nav_L)->default_value(0), "Navigation graph search depth (0 = disabled, 50 = recommended).");
        optional_configs.add_options()("use_laann", po::bool_switch()->default_value(false), "Use LAANN search implementation instead of PageANN (default: false).");
        optional_configs.add_options()("use_look_ahead_search", po::value<bool>(&use_look_ahead_search)->default_value(true), "Enable look-ahead search mode (default: true). Pass false to disable.");
        optional_configs.add_options()("use_pipeline", po::value<bool>(&use_pipeline)->default_value(true), "Enable async I/O pipeline (default: true). Pass false to disable.");
        optional_configs.add_options()("persistence_window_width", po::value<uint32_t>(&persistence_window_width)->default_value(0),
                                       "Persistence check window: scan top-N unvisited nodes each round to test if first-skipped node is prominent. 0 = always look-ahead (default: 0).");
        optional_configs.add_options()("retset_capacity_ratio", po::value<float>(&retset_capacity_ratio)->default_value(2.0f),
                                       "Retset capacity multiplier relative to L (pipeline+cache mode only). Default: 2.0.");
        optional_configs.add_options()("stable_rank_threshold", po::value<uint32_t>(&stable_rank_threshold)->default_value(10),
                                       "Rank position monitored for convergence; clamped to K. 0 = use K (default: 10).");
        optional_configs.add_options()("beamwidth_spike_ratio", po::value<float>(&beamwidth_spike_ratio)->default_value(0.5f),
                                       "Initial spiked beamwidth as fraction of L at convergence (L * ratio). Default: 0.5.");
        optional_configs.add_options()("beam_decay_ratio", po::value<float>(&beam_decay_ratio)->default_value(0.9f),
                                       "Beam width decay per round after convergence spike, floor at W. Default: 0.9.");
        optional_configs.add_options()("cache_order_file", po::value<std::string>(&cache_order_file)->default_value(std::string("")),
                                       "Path to page-frequency order file for non-reordered indexes (LAANN only). "
                                       "If set, pages are loaded by order file via random seeks instead of sequential read. "
                                       "Format: [num_pages uint32][page_id uint32, freq uint32]...");
        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (stable_rank_threshold == 0) stable_rank_threshold = K;
        stable_rank_threshold = std::min(stable_rank_threshold, K);

        // Validate input ratios and thresholds
        uint32_t min_L = *std::min_element(Lvec.begin(), Lvec.end());
        if (cache_ratio < 0.0f || cache_ratio > 1.0f) {
            std::cerr << "Error: cache_ratio must be in [0, 1]. Got: " << cache_ratio << std::endl;
            return -1;
        }
        if (retset_capacity_ratio < 1.0f) {
            std::cerr << "Error: retset_capacity_ratio must be >= 1. Got: " << retset_capacity_ratio << std::endl;
            return -1;
        }
        if (stable_rank_threshold == 0 || stable_rank_threshold >= min_L) {
            std::cerr << "Error: stable_rank_threshold must be > 0 and < L (min L=" << min_L << "). Got: " << stable_rank_threshold << std::endl;
            return -1;
        }
        if (beamwidth_spike_ratio <= 0.0f || beamwidth_spike_ratio > 1.0f) {
            std::cerr << "Error: beamwidth_spike_ratio must be in (0, 1]. Got: " << beamwidth_spike_ratio << std::endl;
            return -1;
        }
        if (beam_decay_ratio <= 0.0f || beam_decay_ratio > 1.0f) {
            std::cerr << "Error: beam_decay_ratio must be in (0, 1]. Got: " << beam_decay_ratio << std::endl;
            return -1;
        }

        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
        if (vm["use_laann"].as<bool>())
            use_laann = true;

        std::cout << "Use LAANN: " << std::boolalpha << use_laann << std::endl;
        std::cout << "Use Look-Ahead Search: " << std::boolalpha << use_look_ahead_search << std::endl;
        std::cout << "Use Pipeline: " << std::boolalpha << use_pipeline << std::endl;
        std::cout << "Persistence Window: " << persistence_window_width << std::endl;

    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            std::cerr << "Filter are currently not supported yet." << std::endl;
            return -1;
        }
        else
        {
            if (data_type == std::string("float")){
                if (use_laann){
                    return search_laann_disk_index<float>(
                        metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K,
                        W, use_look_ahead_search, use_pipeline,
                        persistence_window_width,
                        num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio, retset_capacity_ratio, stable_rank_threshold, beamwidth_spike_ratio, beam_decay_ratio, cache_order_file);
                }else{
                    return search_pageann_disk_index<float>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio);
                }
            }
            else if (data_type == std::string("int8")){
                if (use_laann){
                    return search_laann_disk_index<int8_t>(
                        metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K,
                        W, use_look_ahead_search, use_pipeline,
                        persistence_window_width,
                        num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio, retset_capacity_ratio, stable_rank_threshold, beamwidth_spike_ratio, beam_decay_ratio, cache_order_file);
                }else{
                    return search_pageann_disk_index<int8_t>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio);
                }
            }
            else if (data_type == std::string("uint8")){
                if (use_laann){
                    return search_laann_disk_index<uint8_t>(
                        metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K,
                        W, use_look_ahead_search, use_pipeline,
                        persistence_window_width,
                        num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio, retset_capacity_ratio, stable_rank_threshold, beamwidth_spike_ratio, beam_decay_ratio, cache_order_file);
                }else{
                    return search_pageann_disk_index<uint8_t>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, nav_L, cache_ratio);
                }
            }
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
