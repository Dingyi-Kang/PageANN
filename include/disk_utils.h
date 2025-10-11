// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Page Graph Construction and Disk Layout Header
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"

#include "utils.h"
#include "abstract_graph_store.h"
#include "abstract_data_store.h"
#include "in_mem_graph_store.h"
#include "in_mem_data_store.h"
#include "ooc_in_mem_graph_store.h"
#include "ooc_in_mem_data_store.h"
#include "windows_customizations.h"

namespace diskann
{
const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
const double PQ_TRAINING_SET_FRACTION = 0.1;
const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
const uint32_t NUM_NODES_TO_CACHE = 250000;
const uint32_t WARMUP_L = 20;
const uint32_t NUM_KMEANS_REPS = 12;

template <typename T, typename LabelT> class PQFlashIndex;

DISKANN_DLLEXPORT double get_memory_budget(const std::string &mem_budget_str);
DISKANN_DLLEXPORT double get_memory_budget(double search_ram_budget_in_gb);
DISKANN_DLLEXPORT void add_new_file_to_single_index(std::string index_file, std::string new_file);

DISKANN_DLLEXPORT size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

DISKANN_DLLEXPORT void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs);

#ifdef EXEC_ENV_OLS
template <typename T>
DISKANN_DLLEXPORT T *load_warmup(MemoryMappedFiles &files, const std::string &cache_warmup_file, uint64_t &warmup_num,
                                 uint64_t warmup_dim, uint64_t warmup_aligned_dim);
#else
template <typename T>
DISKANN_DLLEXPORT T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num, uint64_t warmup_dim,
                                 uint64_t warmup_aligned_dim);
#endif

DISKANN_DLLEXPORT int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix,
                                   const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                   const uint64_t nshards, uint32_t max_degree, const std::string &output_vamana,
                                   const std::string &medoids_file, bool use_filters = false,
                                   const std::string &labels_to_medoids_file = std::string(""));

DISKANN_DLLEXPORT void extract_shard_labels(const std::string &in_label_file, const std::string &shard_ids_bin,
                                            const std::string &shard_label_file);

template <typename T>
DISKANN_DLLEXPORT std::string preprocess_base_file(const std::string &infile, const std::string &indexPrefix,
                                                   diskann::Metric &distMetric);

template <typename T, typename LabelT = uint32_t>
DISKANN_DLLEXPORT int build_merged_vamana_index(std::string base_file, std::string index_prefix_path, diskann::Metric _compareMetric, uint32_t L,
                                                uint32_t R, double sampling_rate, double ram_budget,
                                                std::string mem_index_path, std::string medoids_file,
                                                std::string centroids_file, size_t build_pq_bytes, bool use_opq,
                                                uint32_t num_threads, bool use_filters = false,
                                                const std::string &universal_label = "", const uint32_t Lf = 0, const uint32_t num_pq_chunk = 12, const uint64_t page_size = 4);

template <typename T, typename LabelT>
DISKANN_DLLEXPORT uint32_t optimize_beamwidth(std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> &_pFlashIndex,
                                              T *tuning_sample, uint64_t tuning_sample_num,
                                              uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                              uint32_t start_bw = 2);

template <typename T, typename LabelT = uint32_t>
DISKANN_DLLEXPORT int build_disk_index(
    const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
    diskann::Metric _compareMetric, bool use_opq = false,
    const std::string &codebook_prefix = "", // default is empty for no codebook pass in
    bool use_filters = false,
    const std::string &label_file = std::string(""), // default is empty string for no label_file
    const std::string &universal_label = "", const uint32_t filter_threshold = 0,
    const uint32_t Lf = 0); // default is empty string for no universal label

/**
 * @brief Generate a page-level graph index from a Vamana vector-level index.
 *
 * Transforms a traditional Vamana graph into PageANN's page-level organization where
 * multiple vectors are merged into disk-aligned pages. This reduces random I/O during
 * search by co-locating vectors that are likely to be accessed together.
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 *
 * @param index_prefix_path Path prefix for input Vamana index files (graph + metadata)
 * @param data_file_to_use Path to raw vector dataset in binary format
 * @param min_degree_per_node Target minimum degree per node in output page graph
 * @param R Maximum degree of input Vamana graph (used for memory allocation)
 * @param num_pq_chunks_32 Number of Product Quantization chunks for compression
 * @param compareMetric Distance metric (L2, INNER_PRODUCT, or COSINE)
 * @param memBudgetInGB Memory budget in GB for the searching process (affects PQ caching)
 * @param full_ooc If true, use fully out-of-core mode with minimal in-memory structures
 *
 * @return int Returns 0 on success, -1 on failure
 *
 * @note This function writes output files with the same prefix as index_prefix_path
 *       with additional extensions for page graph structure and metadata
 */
template <typename T>
DISKANN_DLLEXPORT int build_page_graph(const std::string &index_prefix_path,
    const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t R, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, float memBudgetInGB, bool full_ooc);

/**
 * @brief Recommend optimal Vamana graph degree for page graph construction.
 *
 * Analyzes dataset characteristics and memory constraints to suggest the best
 * Vamana graph degree (R) for building an index that will be converted to PageANN.
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param data_file_to_use Path to raw vector dataset
 * @param min_degree_per_node Minimum average degree per vector in page graph
 * @param num_pq_chunks_32 Number of PQ chunks for compression
 * @param memBudgetInGB Memory budget in GB for search process
 * @param full_ooc If true, assume fully out-of-core mode
 */
template <typename T>
DISKANN_DLLEXPORT void optimal_vamana_graph_degree(const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t num_pq_chunks_32, float memBudgetInGB, bool full_ooc);

/**
 * @brief Merge vectors into pages using graph-topology-based clustering.
 *
 * Groups vectors that are topologically close in the Vamana graph into pages
 * to minimize random I/O during search operations.
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param nnodes_per_sector Maximum vectors per page
 * @param graph_store Loaded Vamana graph topology
 * @param data_store Loaded raw vector dataset
 * @param npts Total number of vectors
 * @param ndim Vector dimensionality
 * @param index_prefix_path Output path prefix
 * @param R Maximum degree of Vamana graph
 * @param num_pq_chunk Number of PQ chunks
 * @param mergedNodes [Output] Page-to-vectors mapping
 * @param nodeToPageMap [Output] Vector-to-page mapping
 * @param new_to_original_map [Output] New to original ID mapping
 * @return Number of pages created
 */
template <typename T>
DISKANN_DLLEXPORT size_t mergeNodesIntoPage(const uint64_t nnodes_per_sector, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<T>> data_store, const uint32_t npts, const uint32_t ndim, const std::string index_prefix_path,
                                            const uint32_t R, const uint32_t num_pq_chunk, std::vector<std::vector<uint32_t>>& mergedNodes, std::vector<uint32_t>& nodeToPageMap, std::vector<uint32_t>& new_to_original_map);

/**
 * @brief Create disk layout for page-based index storage.
 *
 * Writes the page-level graph index to disk with optimized layout for
 * sequential page access during search.
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param nnodes_per_sector Vectors per page
 * @param pq_cache_ratio Fraction of PQ data to cache
 * @param graph_store Input Vamana graph
 * @param data_store Input vector dataset
 * @param ndims Vector dimensionality
 * @param page_graph_degree Maximum page degree
 * @param pq_compressed_all_nodes_path Input PQ data path
 * @param output_file Output index file path
 * @param mergedNodes Page-to-vectors mapping
 * @param nodeToPageMap Vector-to-page mapping
 */
template <typename T>
DISKANN_DLLEXPORT void create_disk_layout(const uint64_t nnodes_per_sector, const float pq_cache_ratio, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<T>> data_store, const size_t ndims, const uint32_t page_graph_degree, const std::string &pq_compressed_all_nodes_path,
                                          const std::string &output_file, const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToPageMap);


template <typename T>
DISKANN_DLLEXPORT void diskann_create_disk_layout(const std::string base_file, const std::string mem_index_file,
                                        const std::string output_file,
                                        const std::string reorder_data_file = std::string(""));

/**
 * @brief Generate LSH (Locality-Sensitive Hashing) index for fast routing.
 *
 * Creates a hash-based index for quickly identifying candidate entry points
 * during search. Currently experimental and not fully supported.
 *
 * @tparam T Data type of vectors
 * @param data_store Input vector dataset
 * @param num_lsh_sample_nodes Number of vectors to sample for LSH
 * @param dim Vector dimensionality
 * @param reorder_pq_data_buff PQ data buffer
 * @param cached_PQ_nodes Set of cached node IDs
 * @param num_pq_chunks_32 Number of PQ chunks
 * @param index_prefix_path Output path prefix
 * @param new_to_original_map ID mapping
 * @return Size of hash buckets in bytes
 */
template <typename T>
size_t generate_hash_buckets(std::shared_ptr<InMemOOCDataStore<T>> data_store, const uint32_t num_hash_sample_nodes, const size_t dim,  std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint32_t num_pq_chunks_32, const std::string &index_prefix_path, std::vector<uint32_t>& new_to_original_map);

/**
 * @brief Generate reorganized PQ data file and calculate memory usage.
 *
 * @param pq_cache_ratio Target PQ cache ratio
 * @param points_num Total number of vectors
 * @param reorder_pq_data_buff Reordered PQ data buffer
 * @param cached_PQ_nodes Set of cached node IDs
 * @param num_pq_chunks_32 Number of PQ chunks
 * @param pq_compressed_reorder_path Output PQ file path
 * @return Memory usage in bytes
 */
size_t generate_new_pq_data(const float pq_cache_ratio, const size_t points_num, std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint32_t num_pq_chunks_32, const std::string &pq_compressed_reorder_path);

/**
 * @brief Calculate optimal memory allocation ratios for LSH and PQ caching.
 *
 * @param memBudgetInGB Available memory budget in GB
 * @param PQ_size Bytes per PQ-compressed vector
 * @param npts Total number of vectors
 * @return Pair of (lsh_sample_ratio, pq_cache_ratio)
 */
std::pair<float, float> getCacheRatio(float memBudgetInGB, size_t PQ_size, size_t npts);

/**
 * @brief Find optimal page degree and vectors-per-page.
 *
 * @param dimension Vector dimensionality
 * @param typeByte Size of data type
 * @param PQ_size Bytes per PQ-compressed vector
 * @param sampleRatio Cached neighbor ratio
 * @param min_degree_per_node Minimum average degree per vector
 * @return Pair of (page_degree, nnodes_per_page)
 */
std::pair<size_t, size_t> get_optimal_page_degree_nnodes_per_page(size_t dimension, size_t typeByte, size_t PQ_size, float sampleRatio, size_t min_degree_per_node);

/**
 * @brief Calculate how many vectors fit in a page given page degree.
 *
 * @param dimension Vector dimensionality
 * @param typeByte Size of data type
 * @param pageDegree Page outgoing edge count
 * @param PQ_size Bytes per PQ-compressed vector
 * @param sampleRatio Cached neighbor ratio
 * @return Number of vectors per page
 */
size_t get_nnodes_per_page(size_t dimension, size_t typeByte, size_t pageDegree, size_t PQ_size, float sampleRatio);
} // namespace diskann
