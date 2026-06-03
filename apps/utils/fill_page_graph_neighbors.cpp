// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Fill Spare Space in Page-Level Graph
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.
//
// This utility reads an existing PageANN disk index and fills the spare space
// in each page with close neighbors of vectors, starting from the first vector
// (page representative) and proceeding to subsequent vectors until the page is full.

#include <omp.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <set>
#include <limits>
#include <iomanip>

#include "index.h"
#include "utils.h"
#include "index_factory.h"
#include "parameters.h"
#include "defaults.h"
#include "tsl/robin_set.h"
#include "neighbor.h"
#include "ooc_in_mem_graph_store.h"
#include "ooc_in_mem_data_store.h"
#include "filter_utils.h"
#include "cached_io.h"

namespace po = boost::program_options;

/**
 * @brief Search existing Vamana graph to find candidate neighbors for a vector.
 *
 * Performs a greedy beam search starting from the neighbors of the query vector
 * in the existing graph, expanding outward to find more candidates.
 *
 * @tparam T Data type of vectors
 * @param vector_id The vector we're finding candidates for (original ID)
 * @param graph_store Existing Vamana graph
 * @param data_store Data store for distance computation
 * @param L Search list size (larger L = more candidates)
 * @param candidates Output: candidate neighbors with distances (sorted by distance)
 */
template <typename T>
void search_for_candidates(
    const uint32_t vector_id,
    std::shared_ptr<diskann::InMemOOCGraphStore> graph_store,
    std::shared_ptr<diskann::InMemOOCDataStore<T>> data_store,
    const uint32_t L,
    std::vector<diskann::Neighbor>& candidates)
{
    candidates.clear();

    // Use a set to track visited nodes
    tsl::robin_set<uint32_t> visited;
    visited.reserve(L * 2);

    // Mark the query vector as visited to exclude it from candidates
    visited.insert(vector_id);

    // Use NeighborPriorityQueue for correct greedy beam search
    diskann::NeighborPriorityQueue candidate_queue(L);

    // Start from the vector's existing neighbors in Vamana graph
    auto initial_neighbors = graph_store->get_ooc_neighbours(vector_id);

    // Add initial neighbors to candidate queue
    for (uint32_t nbr_id : initial_neighbors)
    {
        if (visited.insert(nbr_id).second)
        {
            float dist = data_store->get_distance(vector_id, nbr_id);
            candidate_queue.insert(diskann::Neighbor(nbr_id, dist));
        }
    }

    // Greedy beam search: always expand closest unexpanded candidate
    while (candidate_queue.has_unexpanded_node())
    {
        // Get closest unexpanded candidate and mark it as expanded
        diskann::Neighbor current = candidate_queue.closest_unexpanded();

        // Expand: get neighbors of current node
        auto nbrs = graph_store->get_ooc_neighbours(current.id);
        for (uint32_t nbr_id : nbrs)
        {
            if (visited.insert(nbr_id).second)
            {
                float dist = data_store->get_distance(vector_id, nbr_id);
                candidate_queue.insert(diskann::Neighbor(nbr_id, dist));
            }
        }
    }

    // Copy results to output vector
    candidates.reserve(candidate_queue.size());
    for (size_t i = 0; i < candidate_queue.size(); i++)
    {
        candidates.push_back(candidate_queue[i]);
    }
}

/**
 * @brief Fill spare space in a single page with close neighbors.
 *
 * Starting from the first vector (page representative), find L closest neighbors
 * and add uninserted ones to fill the spare space. Continue with subsequent
 * vectors until the page is full.
 *
 * @tparam T Data type of vectors
 * @param page_vectors Vector IDs in this page [V0, V1, ..., V_{n-1}] (original IDs)
 * @param existing_neighbors Current neighbors of this page (new IDs)
 * @param graph_store Existing Vamana graph (uses original IDs)
 * @param data_store Data store for distance computation
 * @param old_to_new_map Map from original ID to new (reordered) ID
 * @param L Search list size for candidate discovery
 * @param max_page_neighbors Maximum neighbors allowed per page
 * @param filled_neighbors Output: filled neighbors for this page (new IDs)
 */
template <typename T>
void fill_single_page_neighbors(
    const std::vector<uint32_t>& page_vectors,
    const std::vector<uint32_t>& existing_neighbors,
    std::shared_ptr<diskann::InMemOOCGraphStore> graph_store,
    std::shared_ptr<diskann::InMemOOCDataStore<T>> data_store,
    const std::vector<uint32_t>& old_to_new_map,
    const uint32_t L,
    const uint32_t max_page_neighbors,
    std::vector<uint32_t>& filled_neighbors)
{
    // Start with existing neighbors
    filled_neighbors = existing_neighbors;

    // Calculate spare space
    if (filled_neighbors.size() >= max_page_neighbors)
    {
        // No spare space, nothing to do
        return;
    }

    uint32_t spare_space = max_page_neighbors - static_cast<uint32_t>(filled_neighbors.size());

    // Build set of existing neighbors (new IDs) for fast lookup
    tsl::robin_set<uint32_t> existing_set;
    existing_set.reserve(filled_neighbors.size() + spare_space);
    for (uint32_t nbr_new_id : filled_neighbors)
    {
        existing_set.insert(nbr_new_id);
    }

    // Build set of vectors in this page (original IDs) to avoid self-references
    tsl::robin_set<uint32_t> page_vector_set;
    page_vector_set.reserve(page_vectors.size());
    for (uint32_t vid : page_vectors)
    {
        page_vector_set.insert(vid);
    }

    // Process vectors in order: first vector (page representative) first
    for (size_t vec_idx = 0; vec_idx < page_vectors.size() && spare_space > 0; vec_idx++)
    {
        uint32_t vector_id = page_vectors[vec_idx];  // Original ID

        // Find L closest neighbors for this vector
        std::vector<diskann::Neighbor> candidates;
        search_for_candidates<T>(vector_id, graph_store, data_store, L, candidates);

        // Add closest uninserted neighbors to fill spare space
        for (const auto& candidate : candidates)
        {
            if (spare_space == 0)
                break;

            uint32_t nbr_old_id = candidate.id;

            // Skip if it's a vector in this page
            if (page_vector_set.count(nbr_old_id))
                continue;

            // Convert to new ID
            uint32_t nbr_new_id = old_to_new_map[nbr_old_id];

            // Skip if already in page neighbors
            if (existing_set.count(nbr_new_id))
                continue;

            // Add this neighbor (new ID)
            filled_neighbors.push_back(nbr_new_id);
            existing_set.insert(nbr_new_id);
            spare_space--;
        }
    }
}

/**
 * @brief Main function to fill spare space in PageANN disk index.
 *
 * Reads the existing PageANN disk index, fills spare space in each page,
 * and writes a new filled disk index.
 *
 * @tparam T Data type of vectors
 * @param data_file Path to vector data file
 * @param vamana_index_prefix_path Prefix for Vamana index files (_disk.index)
 * @param pageann_index_prefix_path Prefix for PageANN index files (.index, _new_to_old_ids_map.bin, _centroids.bin)
 * @param output_prefix_path Prefix for output index files
 * @param L Search list size for candidate discovery
 * @param vamana_R Vamana graph max degree
 * @param num_threads Number of threads
 * @param metric Distance metric
 * @return 0 on success, -1 on failure
 */
template <typename T>
int fill_page_graph_neighbors(
    const std::string& data_file,
    const std::string& vamana_index_prefix_path,
    const std::string& pageann_index_prefix_path,
    const std::string& output_prefix_path,
    const uint32_t L,
    const uint32_t vamana_R,
    const uint32_t num_threads,
    diskann::Metric metric)
{
    // Step 0: Setup
    const bool last_page_only = false;  // Set to false to process all pages

    if (num_threads != 0)
        omp_set_num_threads(num_threads);

    std::cout << "========================================" << std::endl;
    std::cout << "Fill Spare Space in PageANN Disk Index" << std::endl;
    std::cout << "========================================" << std::endl;

    // Step 1: Load data and Vamana graph
    size_t points_num, dim;
    diskann::get_bin_metadata(data_file.c_str(), points_num, dim);

    std::cout << "Loading data and graph..." << std::endl;
    std::cout << "  Data file: " << data_file << std::endl;
    std::cout << "  Points: " << points_num << std::endl;
    std::cout << "  Dimensions: " << dim << std::endl;

    std::string vamana_index_file = vamana_index_prefix_path + "_disk.index";
    std::cout << "  Vamana index: " << vamana_index_file << std::endl;

    auto data_store = diskann::IndexFactory::construct_ooc_datastore<T>(
        diskann::DataStoreStrategy::MEMORY, points_num, dim, metric);
    auto graph_store = diskann::IndexFactory::construct_ooc_graphstore(
        diskann::GraphStoreStrategy::MEMORY, points_num, vamana_R);

    data_store->load(data_file);
    graph_store->set_type_size(sizeof(T));
    graph_store->load(vamana_index_file, points_num);

    std::cout << "Data and Vamana graph loaded successfully." << std::endl;

    // Step 2: Read PageANN disk index metadata
    std::string disk_index_file = pageann_index_prefix_path + ".index";
    std::cout << "Reading PageANN disk index from: " << disk_index_file << std::endl;

    std::ifstream index_reader(disk_index_file, std::ios::binary);
    if (!index_reader.is_open())
    {
        std::cerr << "Error: Cannot open disk index file: " << disk_index_file << std::endl;
        return -1;
    }

    // Read entire metadata sector (first sector) for later copying
    std::vector<char> metadata_sector(diskann::defaults::SECTOR_LEN, 0);
    index_reader.read(metadata_sector.data(), diskann::defaults::SECTOR_LEN);
    index_reader.seekg(0);  // Reset to beginning for parsing

    // Read metadata header (first 8 bytes: nr, nc in bin format)
    uint32_t nr, nc;
    index_reader.read(reinterpret_cast<char*>(&nr), sizeof(uint32_t));
    index_reader.read(reinterpret_cast<char*>(&nc), sizeof(uint32_t));

    // Read metadata values following the format in load_from_separate_paths
    // Format: [npts_aligned][ndims][page_size][medoid][max_node_len][megaNode_capacity][pq_ndims][max_page_degree]
    uint64_t npts_aligned, disk_ndims, page_size, medoid_id, max_node_len;
    uint64_t megaNode_capacity, pq_ndims, max_page_degree;

    index_reader.read(reinterpret_cast<char*>(&npts_aligned), sizeof(uint64_t));     // 0: npts (aligned with megaNode_capacity)
    index_reader.read(reinterpret_cast<char*>(&disk_ndims), sizeof(uint64_t));       // 1: ndims
    index_reader.read(reinterpret_cast<char*>(&page_size), sizeof(uint64_t));        // 2: page_size
    index_reader.read(reinterpret_cast<char*>(&medoid_id), sizeof(uint64_t));        // 3: medoid
    index_reader.read(reinterpret_cast<char*>(&max_node_len), sizeof(uint64_t));     // 4: max_node_len
    index_reader.read(reinterpret_cast<char*>(&megaNode_capacity), sizeof(uint64_t)); // 5: megaNode_capacity
    index_reader.read(reinterpret_cast<char*>(&pq_ndims), sizeof(uint64_t));         // 6: pq_ndims
    index_reader.read(reinterpret_cast<char*>(&max_page_degree), sizeof(uint64_t));  // 7: max_page_degree

    // Calculate number of pages and last page vector count using actual points_num
    uint64_t num_pages = (points_num + megaNode_capacity - 1) / megaNode_capacity;
    uint64_t last_page_vector_count = points_num % megaNode_capacity;
    if (last_page_vector_count == 0)
    {
        last_page_vector_count = megaNode_capacity;
    }

    std::cout << "\nPageANN Disk Index Metadata:" << std::endl;
    std::cout << "  npts_aligned (from index): " << npts_aligned << std::endl;
    std::cout << "  actual points_num: " << points_num << std::endl;
    std::cout << "  ndims: " << disk_ndims << std::endl;
    std::cout << "  medoid page ID: " << medoid_id << std::endl;
    std::cout << "  max_node_len: " << max_node_len << std::endl;
    std::cout << "  megaNode_capacity: " << megaNode_capacity << std::endl;
    std::cout << "  max_page_degree: " << max_page_degree << std::endl;
    std::cout << "  num_pages (calculated): " << num_pages << std::endl;
    std::cout << "  last_page_vector_count: " << last_page_vector_count << std::endl;

    // Calculate layout parameters
    // Page layout: [all vectors including centroid] + [num_cached_nbrs (uint16)] + [num_uncached_nbrs (uint16)] + [nbr_ids (uint32 array)]
    uint64_t bytes_per_vector = disk_ndims * sizeof(T);
    uint64_t vectors_space_per_page = megaNode_capacity * bytes_per_vector;  // All vectors including centroid
    uint64_t nbr_header_size = 2 * sizeof(uint16_t);  // num_cached_nbrs + num_uncached_nbrs
    uint64_t available_nbr_space = diskann::defaults::SECTOR_LEN - vectors_space_per_page - nbr_header_size;
    uint32_t max_page_neighbors = static_cast<uint32_t>(available_nbr_space / sizeof(uint32_t));

    std::cout << "  bytes_per_vector: " << bytes_per_vector << std::endl;
    std::cout << "  vectors_space_per_page: " << vectors_space_per_page << std::endl;
    std::cout << "  max_page_neighbors (calculated): " << max_page_neighbors << std::endl;

    // Step 3: Load ID mapping and build mergedNodes
    std::vector<std::vector<uint32_t>> mergedNodes;
    std::vector<uint32_t> new_to_original_map;
    std::vector<uint32_t> old_to_new_map;

    mergedNodes.reserve(num_pages);
    old_to_new_map.assign(points_num, std::numeric_limits<uint32_t>::max());

    // Load the new-to-old ID mapping file
    std::string new_to_old_ids_map_file = pageann_index_prefix_path + "_new_to_old_ids_map.bin";
    std::cout << "\nReading new id to old id map from: " << new_to_old_ids_map_file << std::endl;
    new_to_original_map = diskann::loadTags(new_to_old_ids_map_file, data_file);

    if (new_to_original_map.empty())
    {
        std::cerr << "Error: Failed to load new_to_old_ids_map from: " << new_to_old_ids_map_file << std::endl;
        return -1;
    }

    std::cout << "Building merged nodes and ID mappings..." << std::endl;

    for (uint32_t megaNodeID = 0; megaNodeID < num_pages; megaNodeID++)
    {
        // Last page may have fewer vectors
        uint32_t num_vectors_in_page = (megaNodeID == num_pages - 1)
            ? static_cast<uint32_t>(last_page_vector_count)
            : static_cast<uint32_t>(megaNode_capacity);

        std::vector<uint32_t> groupedNodes;
        groupedNodes.reserve(num_vectors_in_page);

        for (uint32_t k = 0; k < num_vectors_in_page; k++)
        {
            uint32_t new_nid = megaNodeID * static_cast<uint32_t>(megaNode_capacity) + k;
            uint32_t old_nid = new_to_original_map[new_nid];
            old_to_new_map[old_nid] = new_nid;
            groupedNodes.push_back(old_nid);
        }
        mergedNodes.push_back(groupedNodes);
    }

    std::cout << "Built " << mergedNodes.size() << " pages from ID mapping." << std::endl;

    // Step 4: Process each page - read, fill, and write
    std::cout << "\nFill parameters:" << std::endl;
    std::cout << "  L (search list size): " << L << std::endl;
    std::cout << "  last_page_only: " << (last_page_only ? "true" : "false") << std::endl;

    std::string output_index_file = output_prefix_path + ".index";

    // Sector buffer for data pages
    std::vector<char> sector_buf(diskann::defaults::SECTOR_LEN, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    uint32_t maxDeg = 0, minDeg = std::numeric_limits<uint32_t>::max();
    uint64_t totalDeg = 0;
    uint64_t total_added = 0;

    // Determine starting page
    uint32_t start_page = last_page_only ? (num_pages - 1) : 0;

    if (last_page_only)
    {
        // For last_page_only mode, read from original index, write to output file
        std::cout << "\nFixing last page: " << output_index_file << std::endl;

        // Read last page from ORIGINAL PageANN index (not corrupted output)
        uint64_t last_page_offset = (num_pages) * diskann::defaults::SECTOR_LEN;  // +1 for metadata sector
        index_reader.seekg(last_page_offset);
        index_reader.read(sector_buf.data(), diskann::defaults::SECTOR_LEN);
        index_reader.close();

        // Open output file for in-place update
        std::fstream output_rw(output_index_file, std::ios::binary | std::ios::in | std::ios::out);
        if (!output_rw.is_open())
        {
            std::cerr << "Error: Cannot open output file for in-place update: " << output_index_file << std::endl;
            return -1;
        }

        // Parse existing neighbors
        char* nbr_ptr = sector_buf.data() + vectors_space_per_page;
        uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(nbr_ptr);
        uint16_t num_uncached_nbrs = *reinterpret_cast<uint16_t*>(nbr_ptr + sizeof(uint16_t));
        uint32_t existing_num_nbrs = static_cast<uint32_t>(num_cached_nbrs) + static_cast<uint32_t>(num_uncached_nbrs);

        std::cout << "  Last page existing neighbors: " << existing_num_nbrs << std::endl;

        if (existing_num_nbrs > max_page_neighbors)
        {
            std::cout << "  Capping to max_page_neighbors: " << max_page_neighbors << std::endl;
            existing_num_nbrs = max_page_neighbors;
        }

        uint32_t* existing_nbr_ids = reinterpret_cast<uint32_t*>(nbr_ptr + 2 * sizeof(uint16_t));
        std::vector<uint32_t> existing_neighbors(existing_nbr_ids, existing_nbr_ids + existing_num_nbrs);

        // Fill spare space
        std::vector<uint32_t> filled_neighbors;
        fill_single_page_neighbors<T>(
            mergedNodes[num_pages - 1],
            existing_neighbors,
            graph_store,
            data_store,
            old_to_new_map,
            L,
            max_page_neighbors,
            filled_neighbors
        );

        std::cout << "  Filled neighbors: " << filled_neighbors.size() << std::endl;
        std::cout << "  Added: " << (filled_neighbors.size() - existing_neighbors.size()) << std::endl;

        // Write back
        uint16_t filled_num_cached = static_cast<uint16_t>(filled_neighbors.size());
        uint16_t filled_num_uncached = 0;
        std::memcpy(nbr_ptr, &filled_num_cached, sizeof(uint16_t));
        std::memcpy(nbr_ptr + sizeof(uint16_t), &filled_num_uncached, sizeof(uint16_t));
        std::memcpy(nbr_ptr + 2 * sizeof(uint16_t), filled_neighbors.data(),
                    filled_neighbors.size() * sizeof(uint32_t));

        output_rw.seekp(last_page_offset);
        output_rw.write(sector_buf.data(), diskann::defaults::SECTOR_LEN);
        output_rw.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> fix_time = end_time - start_time;

        std::cout << "\n========================================" << std::endl;
        std::cout << "Last Page Fix Complete!" << std::endl;
        std::cout << "  Output file: " << output_index_file << std::endl;
        std::cout << "  Fix time: " << fix_time.count() << " seconds" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }

    // Full processing mode
    std::cout << "\nWriting filled index to: " << output_index_file << std::endl;

    // Setup output writer
    constexpr size_t WRITE_BLK_SIZE = 64 * 1024 * 1024;  // 64MB buffer
    cached_ofstream output_writer(output_index_file, WRITE_BLK_SIZE);

    // Write metadata sector (copy from input, will be preserved as-is)
    output_writer.write(metadata_sector.data(), diskann::defaults::SECTOR_LEN);

    // Seek to first data page (skip metadata sector)
    index_reader.seekg(diskann::defaults::SECTOR_LEN);

    // Process pages sequentially (read one, process, write one)
    for (uint32_t page_id = 0; page_id < num_pages; page_id++)
    {
        // Read this page from disk (sequential, no seek needed)
        index_reader.read(sector_buf.data(), diskann::defaults::SECTOR_LEN);

        // Parse existing neighbors from page
        // Layout: [vectors (megaNode_capacity slots)] + [num_cached_nbrs (uint16)] + [num_uncached_nbrs (uint16)] + [nbr_ids (uint32 array)]
        // Note: Page layout is FIXED at megaNode_capacity vectors, even for last page
        char* nbr_ptr = sector_buf.data() + vectors_space_per_page;
        uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(nbr_ptr);
        uint16_t num_uncached_nbrs = *reinterpret_cast<uint16_t*>(nbr_ptr + sizeof(uint16_t));
        uint32_t existing_num_nbrs = static_cast<uint32_t>(num_cached_nbrs) + static_cast<uint32_t>(num_uncached_nbrs);

        // Sanity check: cap at max_page_neighbors to prevent reading garbage
        if (existing_num_nbrs > max_page_neighbors)
        {
            std::cerr << "Warning: Page " << page_id << " has " << existing_num_nbrs
                      << " neighbors (max allowed: " << max_page_neighbors << "), capping." << std::endl;
            existing_num_nbrs = max_page_neighbors;
        }

        uint32_t* existing_nbr_ids = reinterpret_cast<uint32_t*>(nbr_ptr + 2 * sizeof(uint16_t));
        std::vector<uint32_t> existing_neighbors(existing_nbr_ids, existing_nbr_ids + existing_num_nbrs);

        // Fill spare space for this page
        std::vector<uint32_t> filled_neighbors;
        fill_single_page_neighbors<T>(
            mergedNodes[page_id],
            existing_neighbors,
            graph_store,
            data_store,
            old_to_new_map,
            L,
            max_page_neighbors,
            filled_neighbors
        );

        // Update statistics
        uint32_t num_added = static_cast<uint32_t>(filled_neighbors.size() - existing_neighbors.size());
        total_added += num_added;
        maxDeg = std::max(maxDeg, static_cast<uint32_t>(filled_neighbors.size()));
        minDeg = std::min(minDeg, static_cast<uint32_t>(filled_neighbors.size()));
        totalDeg += filled_neighbors.size();

        // Write filled neighbors back to sector buffer
        // Layout: [num_cached_nbrs (uint16)] + [num_uncached_nbrs (uint16)] + [nbr_ids (uint32 array)]
        // All filled neighbors are cached (num_uncached_nbrs = 0)
        uint16_t filled_num_cached = static_cast<uint16_t>(filled_neighbors.size());
        uint16_t filled_num_uncached = 0;
        std::memcpy(nbr_ptr, &filled_num_cached, sizeof(uint16_t));
        std::memcpy(nbr_ptr + sizeof(uint16_t), &filled_num_uncached, sizeof(uint16_t));
        std::memcpy(nbr_ptr + 2 * sizeof(uint16_t), filled_neighbors.data(),
                    filled_neighbors.size() * sizeof(uint32_t));

        // Write this page to output
        output_writer.write(sector_buf.data(), diskann::defaults::SECTOR_LEN);

        // Progress reporting
        if (page_id % 10000 == 0)
        {
            const float percent = 100.0f * (page_id + 1) / num_pages;
            std::cout << "\r  Processed " << std::setw(6) << std::fixed << std::setprecision(2)
                      << percent << "% (" << page_id + 1 << "/" << num_pages << " pages)"
                      << std::flush;
        }
    }

    index_reader.close();
    output_writer.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fill_time = end_time - start_time;

    std::cout << "\n\nFilled page neighbor statistics:" << std::endl;
    std::cout << "  Max degree: " << maxDeg << std::endl;
    std::cout << "  Min degree: " << minDeg << std::endl;
    std::cout << "  Avg degree: " << static_cast<double>(totalDeg) / num_pages << std::endl;
    std::cout << "  Total neighbors added: " << total_added << std::endl;
    std::cout << "  Avg neighbors added per page: " << static_cast<double>(total_added) / num_pages << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "PageANN Disk Index Fill Complete!" << std::endl;
    std::cout << "  Output file: " << output_index_file << std::endl;
    std::cout << "  Fill time: " << fill_time.count() << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    std::string data_type, dist_fn, data_file;
    std::string vamana_index_prefix_path, pageann_index_prefix_path, output_prefix_path;
    uint32_t L, vamana_R, num_threads;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("data_type",
                          po::value<std::string>(&data_type)->required(),
                          "data type <int8/uint8/float>");

        desc.add_options()("dist_fn",
                          po::value<std::string>(&dist_fn)->required(),
                          "distance function <l2/mips/cosine>");

        desc.add_options()("data_file",
                          po::value<std::string>(&data_file)->required(),
                          "Path to the vector data file (.bin format)");

        desc.add_options()("vamana_index_prefix_path",
                          po::value<std::string>(&vamana_index_prefix_path)->required(),
                          "Prefix path for Vamana index files (expects _disk.index)");

        desc.add_options()("pageann_index_prefix_path",
                          po::value<std::string>(&pageann_index_prefix_path)->required(),
                          "Prefix path for PageANN index files (expects .index, _new_to_old_ids_map.bin, _centroids.bin)");

        desc.add_options()("output_prefix_path",
                          po::value<std::string>(&output_prefix_path)->required(),
                          "Prefix path for output filled PageANN index files");

        desc.add_options()("L,L",
                          po::value<uint32_t>(&L)->default_value(100),
                          "Search list size for candidate discovery");

        desc.add_options()("vamana_R",
                          po::value<uint32_t>(&vamana_R)->required(),
                          "Max degree of existing Vamana graph");

        desc.add_options()("num_threads,T",
                          po::value<uint32_t>(&num_threads)->default_value(0),
                          "Number of threads (0 = use all available)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Determine metric
    diskann::Metric metric;
    if (dist_fn == "l2")
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == "mips")
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == "cosine")
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cerr << "Error: Unsupported distance function. Use l2/mips/cosine" << std::endl;
        return -1;
    }

    try
    {
        if (data_type == "float")
        {
            return fill_page_graph_neighbors<float>(
                data_file, vamana_index_prefix_path, pageann_index_prefix_path,
                output_prefix_path, L, vamana_R, num_threads, metric);
        }
        else if (data_type == "uint8")
        {
            return fill_page_graph_neighbors<uint8_t>(
                data_file, vamana_index_prefix_path, pageann_index_prefix_path,
                output_prefix_path, L, vamana_R, num_threads, metric);
        }
        else if (data_type == "int8")
        {
            return fill_page_graph_neighbors<int8_t>(
                data_file, vamana_index_prefix_path, pageann_index_prefix_path,
                output_prefix_path, L, vamana_R, num_threads, metric);
        }
        else
        {
            std::cerr << "Error: Unsupported data type. Use float/int8/uint8" << std::endl;
            return -1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
