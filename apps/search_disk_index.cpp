// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Page-based Index Search Tool
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

/**
 * @file search_disk_index.cpp
 * @brief CLI tool to search PageANN page-level disk indexes using approximate nearest neighbor search.
 *
 * This tool performs efficient ANN search on page-organized disk indexes. It supports:
 * - Page-based graph navigation with beam search
 * - Product Quantization (PQ) for fast distance approximation
 * - Intelligent page caching based on sample query analysis
 * - LSH-based entry point selection (optional)
 * - Parallel search with configurable thread count
 * - Multiple search list sizes (L values) for recall/latency trade-offs
 *
 * @usage ./search_disk_index --data_type <float|int8|uint8> --dist_fn <l2|mips|cosine>
 *        --index_path_prefix <index_prefix> --query_file <queries.bin>
 *        -K <recall_at> -L <search_list_sizes>
 *        [--num_pages_to_cache <pages>] [--beamwidth <W>] [--num_threads <T>]
 */

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

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
 * @param use_hash_routing Use in-memory hash-based routing for entry point selection
 * @param use_sampled_hash_routing Use sampled in-memory hash-based routing for entry point selection
 * @param radius Hash search radius for entry point selection
 * @return 0 on success, -1 on failure or insufficient recall
 */
template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix, const std::string &pq_path_prefix,
                      const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_pages_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false, const bool use_hash_routing = false, const bool use_sampled_hash_routing = false, const uint32_t radius = 0)
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
    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix, use_hash_routing, use_sampled_hash_routing, radius);

    if (res != 0)
    {
        return res;
    }

    // ===== STEP 3: Generate cache list and load frequently-accessed pages =====
    std::vector<uint32_t> page_list;
    diskann::cout << "Caching " << num_pages_to_cache << " most frequently visited pages based on sample data." << std::endl;

    // Use sample queries to identify most frequently visited pages
    // This enables intelligent caching by profiling actual search patterns
    std::string pageANN_warmup_query_file = index_path_prefix + "_sample_data.bin";
    if (num_pages_to_cache > 0)
        _pFlashIndex->generate_cache_list_from_sample_queries(
            pageANN_warmup_query_file, Lvec[0], 8, num_pages_to_cache, num_threads, page_list, use_hash_routing);

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

    diskann::cout << "\n" << std::string(120, '=') << std::endl;
    diskann::cout << "SEARCH PERFORMANCE RESULTS" << std::endl;
    diskann::cout << std::string(120, '=') << std::endl;

    diskann::cout << std::left
                  << std::setw(6) << "L"
                  << std::setw(10) << "Beamwidth"
                  << std::setw(12) << "QPS"
                  << std::setw(15) << "Latency (us)"
                  << std::setw(12) << "IO (us)"
                  << std::setw(12) << "Other (us)"
                  << std::setw(10) << "Mean IOs"
                  << std::setw(8) << "Hops";
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
                                                 optimized_beamwidth, use_reorder_data, stats + i, use_hash_routing);
#else
                _pFlashIndex->linux_page_search(query + (i * query_aligned_dim), recall_at, L,
                                               query_result_ids_64.data() + (i * recall_at),
                                               query_result_dists[test_id].data() + (i * recall_at),
                                               optimized_beamwidth, use_reorder_data, stats + i, use_hash_routing);
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

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto io_999 = diskann::get_percentile_stats<float>(
                    stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_cache_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cache_hits; });

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
                      << std::setw(10) << optimized_beamwidth
                      << std::setw(12) << std::fixed << std::setprecision(1) << qps
                      << std::setw(15) << std::fixed << std::setprecision(2) << mean_latency
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_io_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << (mean_latency - mean_io_us)
                      << std::setw(10) << std::fixed << std::setprecision(1) << mean_ios
                      << std::setw(8) << std::fixed << std::setprecision(1) << mean_hops;
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

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, query_file, gt_file, pq_path_prefix, filter_label,
        label_type, query_filters_file, use_hash_routing_str, use_sampled_hash_routing_str;
    uint32_t num_threads, K, W, radius, num_pages_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;
    bool use_hash_routing = false;
    bool use_sampled_hash_routing = false;

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
        optional_configs.add_options()("use_hash_routing", po::value<std::string>(&use_hash_routing_str)->default_value(std::string("false")), "Use in-memory hash-based routing for entry point selection.");
        optional_configs.add_options()("use_sampled_hash_routing", po::value<std::string>(&use_sampled_hash_routing_str)->default_value(std::string("false")), "Use sampled in-memory hash-based routing for entry point selection.");
        optional_configs.add_options()("radius", po::value<uint32_t>(&radius)->default_value(0), "Radius for hash-based entry point search.");
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
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;

        if (use_hash_routing_str == "true" || use_hash_routing_str == "True")
            use_hash_routing = true;

        if (use_sampled_hash_routing_str == "true" || use_sampled_hash_routing_str == "True")
            use_sampled_hash_routing = true;

        // Validate that both hash routing options cannot be true simultaneously
        if (use_hash_routing && use_sampled_hash_routing)
        {
            std::cerr << "Error: use_hash_routing and use_sampled_hash_routing cannot both be true." << std::endl;
            return -1;
        }

        std::cout << "Use hash routing: " << std::boolalpha << use_hash_routing << std::endl;
        std::cout << "Use sampled hash routing: " << std::boolalpha << use_sampled_hash_routing << std::endl;
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
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, pq_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_pages_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, pq_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_pages_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, pq_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_pages_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, use_hash_routing, use_sampled_hash_routing, radius);
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