// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// LAANN: Navigation Hubs Construction
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>

#include "index.h"
#include "utils.h"
#include "index_factory.h"
#include "parameters.h"
#include "defaults.h"
#include "tsl/robin_set.h"
#include "neighbor.h"

namespace po = boost::program_options;

// Greedy search on nav graph to find closest neighbors
// Returns top-L closest nodes (excluding the query node itself)
template <typename T>
void greedy_search_nav_graph(
    uint32_t query_nav_idx,
    const std::vector<std::vector<uint32_t>>& adjacency_lists,
    const T* sampled_data,
    uint64_t ndims,
    uint32_t L,
    std::shared_ptr<diskann::Distance<T>> dist_cmp,
    std::vector<std::pair<float, uint32_t>>& results)
{
    results.clear();

    tsl::robin_set<uint32_t> visited;
    visited.reserve(L * 2);
    diskann::NeighborPriorityQueue candidate_queue(L);

    const T* query_vec = sampled_data + (uint64_t)query_nav_idx * ndims;

    // Start from the query node itself
    candidate_queue.insert(diskann::Neighbor(query_nav_idx, 0.0f));
    visited.insert(query_nav_idx);

    // Greedy beam search
    while (candidate_queue.has_unexpanded_node()) {
        diskann::Neighbor current = candidate_queue.closest_unexpanded();

        for (uint32_t nbr_idx : adjacency_lists[current.id]) {
            if (visited.insert(nbr_idx).second) {
                const T* nbr_vec = sampled_data + (uint64_t)nbr_idx * ndims;
                float dist = dist_cmp->compare(query_vec, nbr_vec, ndims);
                candidate_queue.insert(diskann::Neighbor(nbr_idx, dist));
            }
        }
    }

    // Collect results (excluding query itself)
    results.reserve(candidate_queue.size());
    for (size_t i = 0; i < candidate_queue.size(); i++) {
        if (candidate_queue[i].id != query_nav_idx) {
            results.emplace_back(candidate_queue[i].distance, candidate_queue[i].id);
        }
    }
}

// Generate K evenly distributed page IDs from N total pages.
//
// Formula: selected_page[i] = i * (N-1) / (K-1)   (integer division)
// This spaces K indices evenly across [0, N-1], always including both endpoints.
// Example: K=4, N=10 → [0, 3, 6, 9]
//
// The same formula is inverted in the search code (pq_flash_index_laann.cpp) to
// find the nav_idx for a given page_id in O(1) without a hash map.
std::vector<uint32_t> generate_sampled_page_ids(uint32_t num_total_pages, uint32_t num_sampled_pages) {
    std::vector<uint32_t> selected;
    if (num_total_pages == 0 || num_sampled_pages == 0) return selected;
    num_sampled_pages = std::min(num_sampled_pages, num_total_pages);
    selected.reserve(num_sampled_pages);
    if (num_sampled_pages == 1) { selected.push_back(0); return selected; }
    for (uint32_t i = 0; i < num_sampled_pages; ++i) {
        // Cast to uint64_t to avoid overflow for large page counts
        uint32_t page_id = (uint64_t)i * (num_total_pages - 1) / (num_sampled_pages - 1);
        selected.push_back(page_id);
    }
    return selected;
}

// Generate sampling indices within a page using priority order:
// top (0), last (capacity-1), second-to-last (capacity-2), ...
std::vector<uint32_t> generate_sample_indices(uint32_t page_capacity, uint32_t samples_per_page) {
    std::vector<uint32_t> indices;
    indices.reserve(samples_per_page);

    if (samples_per_page == 0 || page_capacity == 0) return indices;

    // First sample: top vector (index 0)
    indices.push_back(0);
    if (indices.size() >= samples_per_page) return indices;

    // Remaining samples: last, second-to-last, third-to-last, ...
    for (uint32_t i = 1; i < page_capacity && indices.size() < samples_per_page; ++i) {
        uint32_t idx = page_capacity - i;  // capacity-1, capacity-2, ...
        if (idx != 0) {  // Don't duplicate index 0
            indices.push_back(idx);
        }
    }

    return indices;
}

// Convert sampled vector index (nav_idx) to original vector ID
// Uses pre-computed sample indices for efficiency
uint32_t sampled_idx_to_original_id(uint32_t sampled_idx, uint32_t samples_per_page,
                                     uint32_t page_capacity, uint32_t num_pages,
                                     const std::vector<uint32_t>& most_page_sample_indices,
                                     const std::vector<uint32_t>& last_page_sample_indices) {
    uint32_t page_id = sampled_idx / samples_per_page;
    uint32_t idx_within_sample = sampled_idx % samples_per_page;

    // Use pre-computed indices (last page may have different sampling)
    const std::vector<uint32_t>& sample_indices = (page_id == num_pages - 1) ? last_page_sample_indices : most_page_sample_indices;

    uint32_t vec_offset_in_page = sample_indices[idx_within_sample];
    return page_id * page_capacity + vec_offset_in_page;
}

// Convert original vector ID to nav_idx (for use during search)
// Returns -1 if the vector is not a sampled nav vector
// sample_indices: the sampling order within a page (e.g., [0, 127, 126, 125, ...])
inline int32_t original_id_to_nav_idx(uint32_t original_id, uint32_t samples_per_page,
                                       uint32_t page_capacity,
                                       const std::vector<uint32_t>& sample_indices) {
    uint32_t page_id = original_id / page_capacity;
    uint32_t vec_offset_in_page = original_id % page_capacity;

    // Find position of vec_offset_in_page in sample_indices
    for (uint32_t i = 0; i < samples_per_page && i < sample_indices.size(); ++i) {
        if (sample_indices[i] == vec_offset_in_page) {
            return static_cast<int32_t>(page_id * samples_per_page + i);
        }
    }
    return -1;  // Not a sampled vector
}

template <typename T>
int build_nav_graph(const std::string &laann_disk_index_file, const uint32_t nav_R,
               const uint32_t nav_L, const float alpha, const uint32_t num_threads,
               const uint32_t samples_per_page, diskann::Metric metric,
               const uint32_t num_sampled_pages = 0, bool skip_build = false)
{
    // Step 1: Read metadata from LAANN disk index (first sector is metadata)
    std::ifstream index_reader(laann_disk_index_file, std::ios::binary);
    if (!index_reader.is_open()) {
        std::cerr << "Error: Cannot open LAANN disk index file: " << laann_disk_index_file << std::endl;
        return -1;
    }

    // Read first sector (4096 bytes) containing metadata
    std::vector<char> metadata_sector(diskann::defaults::SECTOR_LEN);
    index_reader.read(metadata_sector.data(), diskann::defaults::SECTOR_LEN);

    // Parse metadata from sector (LAANN disk index format)
    // First 8 bytes: nr (uint32_t) + nc (uint32_t) header
    // Then uint64_t values: [0]=npts, [1]=ndims, [2]=medoid, [3]=each_node_space, [4]=megaNode_capacity,
    //                       [5]=num_pages, [6]=optimal_vector_degree, [7]=vamana_R
    uint64_t* meta_ptr = reinterpret_cast<uint64_t*>(metadata_sector.data() + 8);  // Skip 8-byte header
    const uint64_t total_vectors = meta_ptr[0];       // npts
    const uint64_t ndims = meta_ptr[1];               // dimensions
    const uint64_t medoid_id = meta_ptr[2];           // medoid
    const uint64_t max_node_len = meta_ptr[3];        // each_node_space = ndims * sizeof(T) (vector stride in page)
    const uint64_t nnodes_per_sector = meta_ptr[4];   // megaNode_capacity (vectors per page)

    const uint32_t meganode_capacity = static_cast<uint32_t>(nnodes_per_sector);
    const uint32_t num_pages = static_cast<uint32_t>((total_vectors + meganode_capacity - 1) / meganode_capacity);
    const uint64_t disk_bytes_per_point = ndims * sizeof(T);

    // Determine sampling mode:
    // - Page subsampling: num_sampled_pages > 0 and < num_pages → sample subset of pages, 1 vector each
    // - Normal: sample samples_per_page vectors from every page
    bool page_subsampling = (num_sampled_pages > 0 && num_sampled_pages < num_pages);
    std::vector<uint32_t> selected_pages;

    uint32_t actual_num_samples_per_page;
    if (page_subsampling) {
        selected_pages = generate_sampled_page_ids(num_pages, num_sampled_pages);
        actual_num_samples_per_page = 1;  // always take top vector from each selected page
        std::cout << "  Page subsampling: " << selected_pages.size() << " pages out of " << num_pages
                  << " (every ~" << num_pages / selected_pages.size() << " pages)" << std::endl;
        std::cout << "  First few selected pages: ";
        for (size_t i = 0; i < std::min((size_t)8, selected_pages.size()); ++i)
            std::cout << selected_pages[i] << " ";
        if (selected_pages.size() > 8) std::cout << "...";
        std::cout << std::endl;
    } else {
        actual_num_samples_per_page = std::min(samples_per_page, meganode_capacity);
        if (actual_num_samples_per_page == 0) {
            std::cerr << "Error: samples_per_page must be at least 1" << std::endl;
            return -1;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "PageANN Disk Index Metadata:" << std::endl;
    std::cout << "  Total vectors: " << total_vectors << std::endl;
    std::cout << "  Dimensions: " << ndims << std::endl;
    std::cout << "  MegaNode capacity: " << meganode_capacity << std::endl;
    std::cout << "  Num pages: " << num_pages << std::endl;
    std::cout << "  Max node len: " << max_node_len << std::endl;
    std::cout << "  Samples per page: " << actual_num_samples_per_page << std::endl;
    std::cout << "========================================" << std::endl;

    // Step 2: Pre-compute sampling indices (avoid repeated calculation in loop)
    // For most pages (full capacity)
    std::vector<uint32_t> most_page_sample_indices = generate_sample_indices(meganode_capacity, actual_num_samples_per_page);

    // For last page (might have fewer vectors)
    uint64_t num_vectors_in_last_page = total_vectors - (uint64_t)(num_pages - 1) * meganode_capacity;
    uint32_t num_samples_in_last_page = std::min(actual_num_samples_per_page, static_cast<uint32_t>(num_vectors_in_last_page));
    std::vector<uint32_t> last_page_sample_indices = generate_sample_indices(static_cast<uint32_t>(num_vectors_in_last_page), num_samples_in_last_page);

    std::cout << "Sampling order within page: ";
    for (size_t i = 0; i < most_page_sample_indices.size() && i < 10; ++i) {
        std::cout << most_page_sample_indices[i] << " ";
    }
    if (most_page_sample_indices.size() > 10) std::cout << "...";
    std::cout << std::endl;

    // Step 3: Read sampled vectors from disk index pages
    uint64_t total_sampled;
    if (page_subsampling) {
        total_sampled = selected_pages.size();  // 1 vector per selected page
    } else {
        total_sampled = (uint64_t)(num_pages - 1) * actual_num_samples_per_page + num_samples_in_last_page;
    }

    diskann::cout << "Reading " << total_sampled << " sampled vectors from "
                  << (page_subsampling ? selected_pages.size() : (size_t)num_pages)
                  << " pages..." << std::endl;

    // 32-byte aligned allocation required: DistanceL2Float uses __builtin_assume_aligned(ptr,32)
    // and _mm256_load_ps (aligned AVX2 load). std::vector<float> only guarantees 4-byte alignment,
    // causing SIGSEGV on the first distance computation in greedy_search_nav_graph.
    // Safe for int8/uint8 too — aligned memory is always valid for unaligned SIMD loads.
    const size_t _sd_bytes = ((total_sampled * ndims * sizeof(T)) + 31) & ~(size_t)31;
    void *_sd_raw = nullptr;
    diskann::alloc_aligned(&_sd_raw, _sd_bytes, 32);
    memset(_sd_raw, 0, _sd_bytes);
    auto _sd_free = [](void *p) { diskann::aligned_free(p); };
    std::unique_ptr<void, decltype(_sd_free)> _sd_owner(_sd_raw, _sd_free);
    T *sampled_data = reinterpret_cast<T *>(_sd_raw);
    std::vector<char> page_buffer(diskann::defaults::SECTOR_LEN);
    uint64_t sampled_idx = 0;

    if (page_subsampling) {
        // Read only selected pages, take top vector (index 0) from each
        for (size_t sel = 0; sel < selected_pages.size(); ++sel) {
            uint32_t page_id = selected_pages[sel];
            uint64_t page_offset = (uint64_t)(page_id + 1) * diskann::defaults::SECTOR_LEN;
            index_reader.seekg(page_offset);
            index_reader.read(page_buffer.data(), diskann::defaults::SECTOR_LEN);

            // Top vector is at offset 0 within the page
            T* src = reinterpret_cast<T*>(page_buffer.data());
            T* dst = sampled_data + sampled_idx * ndims;
            std::memcpy(dst, src, disk_bytes_per_point);
            sampled_idx++;

            if ((sel + 1) % 10000 == 0) {
                diskann::cout << "\r  Processed " << (sel + 1) << " / " << selected_pages.size()
                              << " selected pages" << std::flush;
            }
        }
    } else {
        // Normal mode: iterate all pages, sample multiple vectors per page
        for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
            uint64_t page_offset = (uint64_t)(page_id + 1) * diskann::defaults::SECTOR_LEN;
            index_reader.seekg(page_offset);
            index_reader.read(page_buffer.data(), diskann::defaults::SECTOR_LEN);

            bool is_last_page = (page_id == num_pages - 1);
            const std::vector<uint32_t>& page_sample_indices = is_last_page ? last_page_sample_indices : most_page_sample_indices;
            uint32_t num_samples_this_page = is_last_page ? num_samples_in_last_page : actual_num_samples_per_page;

            for (uint32_t s = 0; s < num_samples_this_page; ++s) {
                uint32_t vec_offset_in_page = page_sample_indices[s];
                char* vec_ptr = page_buffer.data() + vec_offset_in_page * max_node_len;
                T* src = reinterpret_cast<T*>(vec_ptr);
                T* dst = sampled_data + sampled_idx * ndims;
                std::memcpy(dst, src, disk_bytes_per_point);
                sampled_idx++;
            }

            if ((page_id + 1) % 10000 == 0) {
                diskann::cout << "\r  Processed " << (page_id + 1) << " / " << num_pages << " pages" << std::flush;
            }
        }
    }
    diskann::cout << std::endl;
    index_reader.close();

    diskann::cout << "Successfully read " << total_sampled << " sampled vectors" << std::endl;

    // Store sampling info for later ID conversion
    const uint32_t num_sampled = static_cast<uint32_t>(total_sampled);

    // Temp nav graph path — same for both build and skip_build paths
    std::string temp_prefix = laann_disk_index_file.substr(0, laann_disk_index_file.find_last_of('.')) + "_temp_nav";
    std::chrono::duration<double> build_time(0);

    if (!skip_build) {
        // Step 4: Build Vamana navigation graph on sampled vectors
        diskann::cout << "========================================" << std::endl;
        diskann::cout << "Building Vamana navigation graph on sampled vectors..." << std::endl;
        diskann::cout << "  Sampled vectors: " << num_sampled << std::endl;
        diskann::cout << "  R (max degree): " << nav_R << std::endl;
        diskann::cout << "  L (build complexity): " << nav_L << std::endl;
        diskann::cout << "  alpha: " << alpha << std::endl;
        diskann::cout << "  #threads: " << (num_threads == 0 ? omp_get_num_procs() : num_threads) << std::endl;

        // Build parameters
        auto index_build_params = diskann::IndexWriteParametersBuilder(nav_L, nav_R)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        // Determine data type string
        std::string data_type_str;
        if (std::is_same<T, float>::value) data_type_str = "float";
        else if (std::is_same<T, uint8_t>::value) data_type_str = "uint8";
        else if (std::is_same<T, int8_t>::value) data_type_str = "int8";

        // Index configuration
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(ndims)
                          .with_max_points(num_sampled)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type_str)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_pq_dist_build(false)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto nav_index = index_factory.create_instance();

        // Build navigation graph from sampled data
        std::vector<uint32_t> empty_tags;
        auto build_start = std::chrono::high_resolution_clock::now();
        nav_index->build(sampled_data, num_sampled, empty_tags);
        build_time = std::chrono::high_resolution_clock::now() - build_start;

        diskann::cout << "Navigation graph built in " << build_time.count() << " seconds" << std::endl;

        // Step 5: Save navigation graph to temp file
        diskann::cout << "========================================" << std::endl;
        diskann::cout << "Extracting navigation graph structure..." << std::endl;
        nav_index->save(temp_prefix.c_str());
    } else {
        diskann::cout << "========================================" << std::endl;
        diskann::cout << "Skipping Vamana build -- loading existing temp nav graph:" << std::endl;
        diskann::cout << "  " << temp_prefix << std::endl;
        if (!std::ifstream(temp_prefix).good()) {
            std::cerr << "Error: temp nav graph not found: " << temp_prefix << std::endl;
            return -1;
        }
    }

    // Read the temporary graph file
    std::string temp_graph_file = temp_prefix;
    std::ifstream temp_in(temp_graph_file, std::ios::binary);
    if (!temp_in.is_open()) {
        std::cerr << "Error: Cannot open temporary graph file: " << temp_graph_file << std::endl;
        return -1;
    }

    // Read Vamana graph header
    uint64_t file_size_orig;
    uint32_t max_degree_orig, entry_point_orig;
    uint64_t num_frozen_pts;
    temp_in.read((char*)&file_size_orig, sizeof(uint64_t));
    temp_in.read((char*)&max_degree_orig, sizeof(uint32_t));
    temp_in.read((char*)&entry_point_orig, sizeof(uint32_t));
    temp_in.read((char*)&num_frozen_pts, sizeof(uint64_t));

    // Read all adjacency lists (variable format from Vamana)
    std::vector<std::vector<uint32_t>> nav_adjacency_lists;
    size_t bytes_read = sizeof(uint64_t) + 2*sizeof(uint32_t) + sizeof(uint64_t);
    while (bytes_read < file_size_orig) {
        uint32_t num_neighbors;
        temp_in.read((char*)&num_neighbors, sizeof(uint32_t));
        std::vector<uint32_t> neighbors(num_neighbors);
        temp_in.read((char*)neighbors.data(), num_neighbors * sizeof(uint32_t));
        nav_adjacency_lists.push_back(std::move(neighbors));
        bytes_read += sizeof(uint32_t) + num_neighbors * sizeof(uint32_t);
    }
    temp_in.close();

    diskann::cout << "Read " << nav_adjacency_lists.size() << " adjacency lists from navigation graph" << std::endl;

    // Validate that we have adjacency lists for all sampled vectors
    if (nav_adjacency_lists.size() != num_sampled) {
        std::cerr << "Error: Navigation graph has " << nav_adjacency_lists.size()
                  << " nodes but expected " << num_sampled << " sampled vectors" << std::endl;
        return -1;
    }

    // Step 6: Fill nodes with degree < nav_R using greedy search
    diskann::cout << "========================================" << std::endl;
    diskann::cout << "Filling nodes to uniform degree " << nav_R << "..." << std::endl;

    // Create distance comparator for greedy search
    diskann::Metric metric_to_use = metric;
    if (metric == diskann::Metric::COSINE || metric == diskann::Metric::INNER_PRODUCT) {
        if (std::is_floating_point<T>::value) {
            metric_to_use = diskann::Metric::L2;
        }
    }
    std::shared_ptr<diskann::Distance<T>> dist_cmp(diskann::get_distance_function<T>(metric_to_use));

    uint64_t total_neighbors_added = 0;
    uint32_t nodes_filled = 0;
    uint32_t nodes_already_full = 0;

    auto fill_start = std::chrono::high_resolution_clock::now();

    for (int64_t nav_idx = 0; nav_idx < (int64_t)num_sampled; ++nav_idx) {
        uint32_t current_degree = static_cast<uint32_t>(nav_adjacency_lists[nav_idx].size());

        if (current_degree >= nav_R) {
            nodes_already_full++;
            continue;
        }

        // Need to add (nav_R - current_degree) neighbors
        uint32_t neighbors_needed = nav_R - current_degree;

        // Build set of existing neighbors for quick lookup
        tsl::robin_set<uint32_t> existing_neighbors;
        existing_neighbors.reserve(nav_adjacency_lists[nav_idx].size());
        for (uint32_t nbr : nav_adjacency_lists[nav_idx]) {
            existing_neighbors.insert(nbr);
        }

        // Greedy search to find closest neighbors (use 2*nav_R to ensure enough candidates)
        std::vector<std::pair<float, uint32_t>> search_results;
        greedy_search_nav_graph<T>(
            static_cast<uint32_t>(nav_idx),
            nav_adjacency_lists,
            sampled_data,
            ndims,
            2 * nav_R,
            dist_cmp,
            search_results
        );

        // Add new neighbors that aren't already in the adjacency list
        uint32_t added = 0;
        for (const auto& result : search_results) {
            if (added >= neighbors_needed) break;

            uint32_t candidate_idx = result.second;
            if (existing_neighbors.find(candidate_idx) == existing_neighbors.end()) {
                nav_adjacency_lists[nav_idx].push_back(candidate_idx);
                added++;
            }
        }

        if (added > 0) {
            total_neighbors_added += added;
            nodes_filled++;
        }
    }

    auto fill_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fill_time = fill_end - fill_start;

    // Print fill stats
    diskann::cout << "Degree filling complete in " << fill_time.count() << " seconds" << std::endl;
    diskann::cout << "  Nodes already at max degree: " << nodes_already_full << std::endl;
    diskann::cout << "  Nodes filled: " << nodes_filled << std::endl;
    diskann::cout << "  Total neighbors added: " << total_neighbors_added << std::endl;
    if (nodes_filled > 0) {
        diskann::cout << "  Avg neighbors added per filled node: "
                      << (double)total_neighbors_added / nodes_filled << std::endl;
    }

    // Verify uniform degree
    for (uint32_t nav_idx = 0; nav_idx < num_sampled; ++nav_idx) {
        if (nav_adjacency_lists[nav_idx].size() != nav_R) {
            std::cerr << "Error: Node " << nav_idx << " has degree "
                      << nav_adjacency_lists[nav_idx].size() << " instead of " << nav_R << std::endl;
            return -1;
        }
    }
    diskann::cout << "Verified: All nodes have uniform degree " << nav_R << std::endl;

    // Step 7: Translate navigation neighbor IDs from sampled indices to original vector IDs
    diskann::cout << "========================================" << std::endl;
    diskann::cout << "Translating sampled indices to original vector IDs..." << std::endl;

    // Translate sampled index → original vector ID
    // In page subsampling mode: nav_idx directly maps to the top vector of selected_pages[nav_idx]
    // In normal mode: use the existing sampled_idx_to_original_id helper
    auto nav_idx_to_original_id = [&](uint32_t nav_idx) -> uint32_t {
        if (page_subsampling) {
            return selected_pages[nav_idx] * meganode_capacity;  // top vector (offset 0) of selected page
        }
        return sampled_idx_to_original_id(nav_idx, actual_num_samples_per_page, meganode_capacity, num_pages,
                                          most_page_sample_indices, last_page_sample_indices);
    };

    uint32_t entry_point_original_id = nav_idx_to_original_id(entry_point_orig);

    std::vector<std::vector<uint32_t>> translated_adjacency_lists(num_sampled);
    for (uint32_t sampled_idx = 0; sampled_idx < num_sampled; ++sampled_idx) {
        translated_adjacency_lists[sampled_idx].reserve(nav_adjacency_lists[sampled_idx].size());
        for (uint32_t nbr_sampled_idx : nav_adjacency_lists[sampled_idx]) {
            translated_adjacency_lists[sampled_idx].push_back(nav_idx_to_original_id(nbr_sampled_idx));
        }
    }

    // Clean up temp files
    std::remove(temp_graph_file.c_str());
    std::string temp_data_file = temp_prefix + ".data";
    std::remove(temp_data_file.c_str());

    // Step 8: Save navigation graph file with original vector IDs
    diskann::cout << "========================================" << std::endl;
    diskann::cout << "Saving navigation graph file..." << std::endl;

    const std::string nav_graph_file = laann_disk_index_file.substr(0, laann_disk_index_file.find_last_of('.')) + "_nav_graph.index";
    std::ofstream nav_writer(nav_graph_file, std::ios::binary);
    if (!nav_writer.is_open()) {
        std::cerr << "Error: Cannot create navigation graph file: " << nav_graph_file << std::endl;
        return -1;
    }

    // Write metadata: [num_sampled, nav_degree, entry_point_original_id, samples_per_page,
    //                  meganode_capacity, num_total_pages]
    // num_total_pages (field 5): 0 = normal mode (all pages sampled);
    //   >0 = page subsampling mode — search uses generate_sampled_page_ids(num_total_pages, num_sampled)
    //   to reconstruct which pages were selected.
    uint64_t nav_metadata[6];
    nav_metadata[0] = num_sampled;
    nav_metadata[1] = nav_R;
    nav_metadata[2] = entry_point_original_id;
    nav_metadata[3] = actual_num_samples_per_page;
    nav_metadata[4] = meganode_capacity;
    nav_metadata[5] = page_subsampling ? (uint64_t)num_pages : 0;

    nav_writer.write(reinterpret_cast<const char*>(nav_metadata), 6 * sizeof(uint64_t));

    // Write fixed-size neighbor lists (all nodes have nav_R neighbors)
    // Neighbor IDs are original vector IDs
    for (uint32_t nav_idx = 0; nav_idx < num_sampled; ++nav_idx) {
        nav_writer.write(reinterpret_cast<const char*>(translated_adjacency_lists[nav_idx].data()),
                         nav_R * sizeof(uint32_t));
    }

    nav_writer.close();

    std::cout << "========================================" << std::endl;
    std::cout << "Navigation Graph Successfully Built!" << std::endl;
    std::cout << "  Output file: " << nav_graph_file << std::endl;
    std::cout << "  Total vectors in index: " << total_vectors << std::endl;
    std::cout << "  Sampled vectors: " << num_sampled << std::endl;
    std::cout << "  Samples per page: " << actual_num_samples_per_page << std::endl;
    std::cout << "  MegaNode capacity: " << meganode_capacity << std::endl;
    std::cout << "  Num pages: " << num_pages << std::endl;
    std::cout << "  Uniform degree: " << nav_R << std::endl;
    std::cout << "  Entry point (original ID): " << entry_point_original_id << std::endl;
    std::cout << "  Build time: " << build_time.count() << " seconds" << std::endl;
    std::cout << "  Fill time: " << fill_time.count() << " seconds" << std::endl;
    std::cout << "  Neighbors added during fill: " << total_neighbors_added << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, dist_fn, laann_disk_index_file;
    uint32_t nav_R, nav_L, num_threads, samples_per_page, num_sampled_pages;
    float alpha;
    bool skip_build;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("data_type",
                          po::value<std::string>(&data_type)->required(),
                          "data type <int8/uint8/float>");

        desc.add_options()("dist_fn",
                          po::value<std::string>(&dist_fn)->required(),
                          "distance function <l2/mips/cosine>");

        desc.add_options()("laann_disk_index_file",
                          po::value<std::string>(&laann_disk_index_file)->required(),
                          "Path to PageANN disk graph index file");

        desc.add_options()("samples_per_page,S",
                          po::value<uint32_t>(&samples_per_page)->default_value(1),
                          "Number of vectors to sample from each page (1 to meganode_capacity). "
                          "Sampling order: top(0), last, second-to-last, ... Ignored when --num_sampled_pages is set.");

        desc.add_options()("num_sampled_pages",
                          po::value<uint32_t>(&num_sampled_pages)->default_value(0),
                          "Sample this many pages from all pages (evenly distributed). "
                          "Takes 1 vector (top) per selected page. "
                          "0 = disabled (sample all pages using --samples_per_page). "
                          "Use when memory is insufficient to sample one vector per page.");

        desc.add_options()("max_degree,R",
                          po::value<uint32_t>(&nav_R)->default_value(64),
                          "Maximum degree for navigation graph");

        desc.add_options()("Lbuild,L",
                          po::value<uint32_t>(&nav_L)->default_value(100),
                          "Build complexity for navigation graph");

        desc.add_options()("alpha",
                          po::value<float>(&alpha)->default_value(1.2f),
                          "alpha controls density and diameter of graph");

        desc.add_options()("num_threads,T",
                          po::value<uint32_t>(&num_threads)->default_value(0),
                          "Number of threads for building (0 = use all available)");

        desc.add_options()("skip_build",
                          po::bool_switch(&skip_build)->default_value(false),
                          "Skip Vamana build and load existing _temp_nav graph for fill+translate steps");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Determine metric
    diskann::Metric metric;
    if (dist_fn == std::string("l2")) {
        metric = diskann::Metric::L2;
    } else if (dist_fn == std::string("mips")) {
        metric = diskann::Metric::INNER_PRODUCT;
    } else if (dist_fn == std::string("cosine")) {
        metric = diskann::Metric::COSINE;
    } else {
        std::cerr << "Error: Unsupported distance function. Use l2/mips/cosine" << std::endl;
        return -1;
    }

    try {
        if (data_type == std::string("float")) {
            return build_nav_graph<float>(laann_disk_index_file, nav_R, nav_L, alpha, num_threads, samples_per_page, metric, num_sampled_pages, skip_build);
        } else if (data_type == std::string("uint8")) {
            return build_nav_graph<uint8_t>(laann_disk_index_file, nav_R, nav_L, alpha, num_threads, samples_per_page, metric, num_sampled_pages, skip_build);
        } else if (data_type == std::string("int8")) {
            return build_nav_graph<int8_t>(laann_disk_index_file, nav_R, nav_L, alpha, num_threads, samples_per_page, metric, num_sampled_pages, skip_build);
        } else {
            std::cerr << "Error: Unsupported data type. Use float/int8/uint8" << std::endl;
            return -1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        diskann::cerr << "Navigation graph build failed." << std::endl;
        return -1;
    }
}
