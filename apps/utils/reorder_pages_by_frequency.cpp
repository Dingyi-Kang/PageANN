// Reorder Pages by Visit Frequency
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.
//
// This utility:
// 1. Profiles page visit frequency using sample queries
// 2. Reorders pages by frequency (most visited first)
// 3. Builds new ID mappings
// 4. Rewrites the disk index with reordered pages and converted neighbor IDs

#include <omp.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>

#include "partition.h"
#include "pq_flash_index_laann.h"
#include "linux_aligned_file_reader.h"
#include "utils.h"
#include "defaults.h"
#include "cached_io.h"
#include "filter_utils.h"

namespace po = boost::program_options;

template <typename T>
int reorder_pages_by_frequency(
    const std::string& index_path_prefix,
    const std::string& data_bin,
    const float sample_ratio,
    const bool skip_profiling,
    diskann::Metric metric,
    const std::string& orig_pq_compressed_file,
    const std::string& orig_gt_file)
{
    constexpr uint32_t L           = 100;
    constexpr uint32_t beam_width  = 5;
    constexpr uint32_t num_threads = 16;

    const std::string output_prefix = index_path_prefix + "_fsort";

    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::cout << "========================================" << std::endl;
    std::cout << "Reorder Pages by Visit Frequency" << std::endl;
    std::cout << "========================================" << std::endl;

    omp_set_num_threads(num_threads);

    // ========================================
    // Read index metadata (used by multiple steps)
    // ========================================
    const std::string index_file = index_path_prefix + ".index";
    std::vector<char> metadata_sector(diskann::defaults::SECTOR_LEN);
    {
        std::ifstream meta_reader(index_file, std::ios::binary);
        if (!meta_reader)
        {
            std::cerr << "Failed to open index file: " << index_file << std::endl;
            return -1;
        }
        meta_reader.read(metadata_sector.data(), diskann::defaults::SECTOR_LEN);
    }

    char* meta_ptr = metadata_sector.data() + 2 * sizeof(uint32_t);  // skip nr, nc
    // LAANN disk format: [0]=npts, [1]=ndims, [2]=medoid(page_id),
    //                       [3]=each_node_space, [4]=nnodes_per_sector, [5]=num_pages,
    //                       [6]=mega_node_degree, [7]=vamana_R
    uint64_t npts, ndims, medoid_page_id, each_node_space, nnodes_per_sector, num_pages_meta, mega_node_degree, vamana_R;
    std::memcpy(&npts,             meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&ndims,            meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&medoid_page_id,   meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&each_node_space,  meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&nnodes_per_sector,meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&num_pages_meta,   meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&mega_node_degree, meta_ptr, sizeof(uint64_t)); meta_ptr += sizeof(uint64_t);
    std::memcpy(&vamana_R,         meta_ptr, sizeof(uint64_t));

    std::cout << "  Index: npts=" << npts << ", ndims=" << ndims
              << ", nnodes_per_sector=" << nnodes_per_sector
              << ", medoid_page=" << medoid_page_id << std::endl;

    // ========================================
    // Step 1: Load index and run profiling (optional)
    // ========================================
    std::string freq_output_file = output_prefix + "_page_frequency.bin";

    if (!skip_profiling)
    {
        std::cout << "\n=== Step 1: Sample queries and profile page frequencies ===" << std::endl;

        // Generate random sample from data file
        const std::string temp_sample_prefix = output_prefix + "_temp_sample";
        const std::string temp_sample_bin    = temp_sample_prefix + "_data.bin";
        const std::string temp_sample_ids    = temp_sample_prefix + "_ids.bin";

        std::cout << "  Generating random sample (ratio=" << sample_ratio << ")..." << std::endl;
        gen_random_slice<T>(data_bin, temp_sample_prefix, static_cast<double>(sample_ratio));
        std::cout << "  Sample written to: " << temp_sample_bin << std::endl;

        // Load index and profile page frequencies
        std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
        std::unique_ptr<diskann::PQFlashIndexLAANN<T>> pFlashIndex(
            new diskann::PQFlashIndexLAANN<T>(reader, metric));

        int load_result = pFlashIndex->load(num_threads, index_path_prefix.c_str(), index_path_prefix.c_str(), beam_width);
        if (load_result != 0)
        {
            std::cerr << "Failed to load index." << std::endl;
            std::cout << "  Deleting temp sample files..." << std::endl;
            std::remove(temp_sample_bin.c_str());
            std::remove(temp_sample_ids.c_str());
            std::cout << "  Deleted: " << temp_sample_bin << ", " << temp_sample_ids << std::endl;
            return -1;
        }

        std::cout << "  Loaded index: " << pFlashIndex->get_num_points() << " points, "
                  << pFlashIndex->get_num_mega_nodes() << " pages" << std::endl;

        pFlashIndex->profile_page_frequency(temp_sample_bin, L, beam_width, num_threads, freq_output_file, L);
        pFlashIndex.reset();

        std::cout << "  Deleting temp sample files..." << std::endl;
        std::remove(temp_sample_bin.c_str());
        std::remove(temp_sample_ids.c_str());
        std::cout << "  Deleted: " << temp_sample_bin << ", " << temp_sample_ids << std::endl;
        std::cout << "  Page frequencies written to: " << freq_output_file << std::endl;
    }
    else
    {
        std::cout << "\n=== Step 1: Skipped (using existing frequency file) ===" << std::endl;
        std::cout << "  Frequency file: " << freq_output_file << std::endl;
    }

    // ========================================
    // Step 2: Load frequency data and sort pages
    // ========================================
    std::cout << "\n=== Step 2: Sort pages by visit frequency ===" << std::endl;

    // Load the frequency file (format: [num_pages][page_id, frequency] pairs sorted by frequency)
    std::ifstream freq_reader(freq_output_file, std::ios::binary);
    if (!freq_reader)
    {
        std::cerr << "Failed to open frequency file: " << freq_output_file << std::endl;
        return -1;
    }

    // Read header: num_pages (uint32_t)
    uint32_t freq_num_pages;
    freq_reader.read(reinterpret_cast<char*>(&freq_num_pages), sizeof(uint32_t));

    std::cout << "  Frequency file: " << freq_num_pages << " pages" << std::endl;

    // Read page IDs in frequency order (already sorted by profile_page_frequency)
    std::vector<uint32_t> sorted_page_ids(freq_num_pages);

    for (uint32_t i = 0; i < freq_num_pages; i++)
    {
        uint32_t page_id, freq;
        freq_reader.read(reinterpret_cast<char*>(&page_id), sizeof(uint32_t));
        freq_reader.read(reinterpret_cast<char*>(&freq), sizeof(uint32_t));
        sorted_page_ids[i] = page_id;

        // Print top 5
        if (i < 5)
        {
            std::cout << "  Page " << page_id << " (freq: " << freq << ")" << std::endl;
        }
    }
    freq_reader.close();

    // Keep the last page (partial page) at the end to preserve ID alignment
    // Remove last page from sorted list and append it at the end
    uint32_t last_page_id = freq_num_pages - 1;
    auto it = std::find(sorted_page_ids.begin(), sorted_page_ids.end(), last_page_id);
    if (it != sorted_page_ids.end())
    {
        sorted_page_ids.erase(it);
        sorted_page_ids.push_back(last_page_id);
        std::cout << "  Keeping last page " << last_page_id << " at the end (may have fewer vectors)" << std::endl;
    }

    // ========================================
    // Step 3: Load old ID mapping and build old mergedNodes
    // ========================================
    std::cout << "\n=== Step 3: Load existing ID mapping ===" << std::endl;

    std::string curr_to_orig_file = index_path_prefix + "_new_to_old_ids_map.bin";
    std::vector<uint32_t> curr_to_original = diskann::loadTags(curr_to_orig_file, "");

    if (curr_to_original.empty())
    {
        std::cerr << "Failed to load old ID mapping from: " << curr_to_orig_file << std::endl;
        return -1;
    }

    uint64_t num_points = curr_to_original.size();
    uint64_t num_pages = freq_num_pages;

    std::cout << "  Loaded " << num_points << " ID mappings" << std::endl;

    // Build old mergedNodes from old mapping
    std::vector<std::vector<uint32_t>> old_mergedNodes(num_pages);
    for (uint32_t page_id = 0; page_id < num_pages; page_id++)
    {
        old_mergedNodes[page_id].reserve(nnodes_per_sector);
        for (uint32_t k = 0; k < nnodes_per_sector; k++)
        {
            uint32_t curr_id = page_id * nnodes_per_sector + k;
            if (curr_id >= curr_to_original.size())
                break;
            uint32_t orig_id = curr_to_original[curr_id];
            old_mergedNodes[page_id].push_back(orig_id);
        }
    }

    // ========================================
    // Step 4: Build new mergedNodes based on frequency order
    // ========================================
    std::cout << "\n=== Step 4: Build new page ordering ===" << std::endl;

    // Build new mergedNodes: new_mergedNodes[new_page_id] = old_mergedNodes[old_page_id]
    std::vector<std::vector<uint32_t>> new_mergedNodes(num_pages);
    for (size_t new_page_id = 0; new_page_id < num_pages; new_page_id++)
    {
        uint32_t old_page_id = sorted_page_ids[new_page_id];
        new_mergedNodes[new_page_id] = old_mergedNodes[old_page_id];
    }

    // Build new ID mappings
    std::vector<uint32_t> new_to_original;
    std::vector<uint32_t> original_to_new(num_points, UINT32_MAX);
    new_to_original.reserve(num_points);

    for (const auto& page : new_mergedNodes)
    {
        for (uint32_t orig_id : page)
        {
            original_to_new[orig_id] = static_cast<uint32_t>(new_to_original.size());
            new_to_original.push_back(orig_id);
        }
    }

    std::cout << "  New ID mapping size: " << new_to_original.size() << std::endl;

    // ========================================
    // Step 5: Load existing disk index and rewrite
    // ========================================
    std::cout << "\n=== Step 5: Rewrite disk index with new ordering ===" << std::endl;

    std::string new_index_file = output_prefix + ".index";

    // Open old index for reading
    std::ifstream old_index_reader(index_file, std::ios::binary);
    if (!old_index_reader)
    {
        std::cerr << "Failed to open old index: " << index_file << std::endl;
        return -1;
    }
    old_index_reader.seekg(diskann::defaults::SECTOR_LEN);  // skip metadata sector (already read)

    // Update medoid page ID to new ordering
    uint64_t new_medoid_page_id = 0;
    for (size_t i = 0; i < sorted_page_ids.size(); i++)
    {
        if (sorted_page_ids[i] == medoid_page_id)
        {
            new_medoid_page_id = i;
            break;
        }
    }
    std::cout << "    new medoid_page: " << new_medoid_page_id << std::endl;

    // Calculate offsets
    uint64_t vectors_space_per_page = nnodes_per_sector * ndims * sizeof(T);

    // Setup output writer
    constexpr size_t WRITE_BLK_SIZE = 64 * 1024 * 1024;
    cached_ofstream output_writer(new_index_file, WRITE_BLK_SIZE);

    // Update and write metadata sector
    // Update medoid_page_id at position [2] (after npts, ndims) in LAANN format
    meta_ptr = metadata_sector.data() + 2 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
    std::memcpy(meta_ptr, &new_medoid_page_id, sizeof(uint64_t));
    output_writer.write(metadata_sector.data(), diskann::defaults::SECTOR_LEN);

    // Process pages one by one (no need to load all into memory)
    std::vector<char> old_sector(diskann::defaults::SECTOR_LEN);
    std::vector<char> new_sector(diskann::defaults::SECTOR_LEN);
    uint32_t nbrs_converted = 0;

    for (uint32_t new_page_id = 0; new_page_id < num_pages; new_page_id++)
    {
        uint32_t old_page_id = sorted_page_ids[new_page_id];

        // Seek to old page position and read (skip metadata sector)
        uint64_t old_page_offset = (1 + old_page_id) * diskann::defaults::SECTOR_LEN;
        old_index_reader.seekg(old_page_offset);
        old_index_reader.read(old_sector.data(), diskann::defaults::SECTOR_LEN);

        // Copy vectors as-is (they use original IDs, which don't change)
        std::memcpy(new_sector.data(), old_sector.data(), vectors_space_per_page);

        // Read neighbor count from old sector (LAANN format: single uint32_t)
        const char* old_nbr_ptr = old_sector.data() + vectors_space_per_page;
        uint32_t total_nbrs;
        std::memcpy(&total_nbrs, old_nbr_ptr, sizeof(uint32_t));

        // Copy neighbor count and convert neighbor IDs: curr -> original -> new
        char* new_nbr_ptr = new_sector.data() + vectors_space_per_page;
        std::memcpy(new_nbr_ptr, &total_nbrs, sizeof(uint32_t));

        const uint32_t* old_nbr_ids = reinterpret_cast<const uint32_t*>(old_nbr_ptr + sizeof(uint32_t));
        uint32_t* new_nbr_ids = reinterpret_cast<uint32_t*>(new_nbr_ptr + sizeof(uint32_t));

        for (uint32_t i = 0; i < total_nbrs; i++)
        {
            uint32_t curr_id = old_nbr_ids[i];
            // Convert: curr_id -> original_id -> new_id
            uint32_t orig_id = curr_to_original[curr_id];
            uint32_t new_id = original_to_new[orig_id];
            new_nbr_ids[i] = new_id;
            nbrs_converted++;
        }

        // Clear remaining sector space
        size_t nbr_data_size = sizeof(uint32_t) + total_nbrs * sizeof(uint32_t);
        size_t remaining = diskann::defaults::SECTOR_LEN - vectors_space_per_page - nbr_data_size;
        if (remaining > 0)
        {
            std::memset(new_nbr_ptr + nbr_data_size, 0, remaining);
        }

        output_writer.write(new_sector.data(), diskann::defaults::SECTOR_LEN);

        if (new_page_id % 100000 == 0)
        {
            std::cout << "\r  Processing: " << new_page_id << " / " << num_pages << " pages" << std::flush;
        }
    }
    std::cout << "\r  Processing: " << num_pages << " / " << num_pages << " pages" << std::endl;

    old_index_reader.close();
    output_writer.close();
    std::cout << "  Converted " << nbrs_converted << " neighbor IDs" << std::endl;

    // ========================================
    // Step 6: Reorder navigation graph
    // ========================================
    std::cout << "\n=== Step 6: Reorder navigation graph ===" << std::endl;

    std::string old_nav_graph_file = index_path_prefix + "_nav_graph.index";
    std::string new_nav_graph_file = output_prefix + "_nav_graph.index";

    std::ifstream nav_graph_reader(old_nav_graph_file, std::ios::binary);
    if (!nav_graph_reader)
    {
        std::cerr << "Warning: Navigation graph not found: " << old_nav_graph_file << std::endl;
        std::cerr << "  Skipping navigation graph reordering." << std::endl;
    }
    else
    {
        // LAANN nav graph format: 5 × uint64_t header
        // [num_sampled, nav_R (uniform degree), entry_point_id, samples_per_page, meganode_capacity]
        // Per-node: fixed nav_R neighbor IDs (uint32_t each), no per-node count field
        uint64_t nav_metadata[5];
        nav_graph_reader.read(reinterpret_cast<char*>(nav_metadata), 5 * sizeof(uint64_t));

        uint32_t nav_num_sampled    = static_cast<uint32_t>(nav_metadata[0]);
        uint32_t nav_uniform_degree = static_cast<uint32_t>(nav_metadata[1]);
        uint32_t old_entry_point_id = static_cast<uint32_t>(nav_metadata[2]);  // old new_id
        uint32_t samples_per_page   = static_cast<uint32_t>(nav_metadata[3]);
        uint32_t nav_meganode_capacity = static_cast<uint32_t>(nav_metadata[4]);

        std::cout << "  Nav graph: " << nav_num_sampled << " sampled nodes, uniform degree " << nav_uniform_degree << std::endl;
        std::cout << "  Samples per page: " << samples_per_page << std::endl;
        std::cout << "  Old entry point (new_id): " << old_entry_point_id << std::endl;

        // Read old adjacency lists (fixed uniform degree, neighbors are old new_ids)
        std::vector<std::vector<uint32_t>> old_adj_lists(nav_num_sampled);
        for (uint32_t i = 0; i < nav_num_sampled; i++)
        {
            old_adj_lists[i].resize(nav_uniform_degree);
            nav_graph_reader.read(reinterpret_cast<char*>(old_adj_lists[i].data()),
                                  nav_uniform_degree * sizeof(uint32_t));
        }
        nav_graph_reader.close();

        // Reorder adjacency lists based on page frequency order
        // nav_idx = page_id * samples_per_page + position_in_page
        // NOTE: This assumes all pages have the same samples_per_page. If the last page
        // has fewer samples, this logic may be incorrect. For most cases (samples_per_page=1),
        // this is not an issue.
        std::vector<std::vector<uint32_t>> new_adj_lists(nav_num_sampled);
        for (uint32_t new_nav_idx = 0; new_nav_idx < nav_num_sampled; new_nav_idx++)
        {
            uint32_t new_page_id = new_nav_idx / samples_per_page;
            uint32_t position_in_page = new_nav_idx % samples_per_page;
            uint32_t old_page_id = sorted_page_ids[new_page_id];
            uint32_t old_nav_idx = old_page_id * samples_per_page + position_in_page;

            const auto& old_neighbors = old_adj_lists[old_nav_idx];
            new_adj_lists[new_nav_idx].resize(old_neighbors.size());

            // Convert neighbor IDs: curr_id -> original_id -> new_id
            for (size_t j = 0; j < old_neighbors.size(); j++)
            {
                uint32_t curr_id = old_neighbors[j];
                uint32_t orig_id = curr_to_original[curr_id];
                uint32_t new_id = original_to_new[orig_id];
                new_adj_lists[new_nav_idx][j] = new_id;
            }
        }

        // Convert entry point: curr_id -> original_id -> new_id
        uint32_t entry_orig_id = curr_to_original[old_entry_point_id];
        uint32_t new_entry_point_id = original_to_new[entry_orig_id];
        std::cout << "  New entry point (new_id): " << new_entry_point_id << std::endl;

        // Write new navigation graph
        std::ofstream nav_graph_writer(new_nav_graph_file, std::ios::binary);

        // Update entry point at position [2] and write 5-field header
        nav_metadata[2] = new_entry_point_id;
        nav_graph_writer.write(reinterpret_cast<char*>(nav_metadata), 5 * sizeof(uint64_t));

        // Write adjacency lists (fixed uniform degree, no per-node count field)
        for (uint32_t nav_idx = 0; nav_idx < nav_num_sampled; nav_idx++)
        {
            nav_graph_writer.write(reinterpret_cast<const char*>(new_adj_lists[nav_idx].data()),
                                   nav_uniform_degree * sizeof(uint32_t));
        }
        nav_graph_writer.close();
        std::cout << "  Written: " << new_nav_graph_file << std::endl;
    }

    // ========================================
    // Step 7: Write new ID mapping files
    // ========================================
    std::cout << "\n=== Step 7: Write new ID mapping files ===" << std::endl;

    // Write new_to_old map
    std::string new_to_old_file = output_prefix + "_new_to_old_ids_map.bin";
    std::ofstream new_to_old_writer(new_to_old_file, std::ios::binary);
    int32_t npts_i32 = static_cast<int32_t>(new_to_original.size());
    int32_t dim_i32 = 1;
    new_to_old_writer.write(reinterpret_cast<char*>(&npts_i32), sizeof(int32_t));
    new_to_old_writer.write(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
    new_to_old_writer.write(reinterpret_cast<char*>(new_to_original.data()), new_to_original.size() * sizeof(uint32_t));
    new_to_old_writer.close();
    std::cout << "  Written: " << new_to_old_file << std::endl;

    // Write old_to_new map
    std::string old_to_new_file = output_prefix + "_old_to_new_ids_map.bin";
    std::ofstream old_to_new_writer(old_to_new_file, std::ios::binary);
    npts_i32 = static_cast<int32_t>(original_to_new.size());
    old_to_new_writer.write(reinterpret_cast<char*>(&npts_i32), sizeof(int32_t));
    old_to_new_writer.write(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
    old_to_new_writer.write(reinterpret_cast<char*>(original_to_new.data()), original_to_new.size() * sizeof(uint32_t));
    old_to_new_writer.close();
    std::cout << "  Written: " << old_to_new_file << std::endl;

    // ========================================
    // Step 8: Reorder PQ data
    // ========================================
    {
        std::cout << "\n=== Step 8: Reorder PQ data ===" << std::endl;

        std::ifstream pq_reader(orig_pq_compressed_file, std::ios::binary);
        if (!pq_reader)
        {
            std::cerr << "Failed to open PQ compressed file: " << orig_pq_compressed_file << std::endl;
            return -1;
        }
        {
            // Input: standard DiskANN PQ format [int32 npts][int32 ndims][data]
            int all_pq_npts, pq_ndims_val;
            pq_reader.read(reinterpret_cast<char*>(&all_pq_npts), sizeof(int));
            pq_reader.read(reinterpret_cast<char*>(&pq_ndims_val), sizeof(int));
            std::cout << "  PQ data: " << all_pq_npts << " points, " << pq_ndims_val << " chunks" << std::endl;

            const size_t pq_npts = static_cast<size_t>(all_pq_npts);
            const size_t pq_dims = static_cast<size_t>(pq_ndims_val);
            const size_t total_pq_bytes = pq_npts * pq_dims;

            std::unique_ptr<uint8_t[]> pq_data = std::make_unique<uint8_t[]>(total_pq_bytes);
            pq_reader.read(reinterpret_cast<char*>(pq_data.get()), total_pq_bytes);
            pq_reader.close();

            // Reorder: reordered_pq[new_id] = pq_data[original_id]
            std::unique_ptr<uint8_t[]> reordered_pq = std::make_unique<uint8_t[]>(total_pq_bytes);
            for (size_t new_id = 0; new_id < new_to_original.size(); new_id++)
            {
                uint32_t orig_id = new_to_original[new_id];
                std::memcpy(&reordered_pq[new_id * pq_dims], &pq_data[orig_id * pq_dims], pq_dims);
            }
            std::cout << "  Reordered " << new_to_original.size() << " PQ vectors" << std::endl;

            // Output: LAANN PQ format [uint32 npts][uint32 num_chunks][data]
            std::string reorder_pq_out = output_prefix + "_reorder_pq_compressed.bin";
            std::ofstream pq_writer(reorder_pq_out, std::ios::binary);
            uint32_t points_num_32   = static_cast<uint32_t>(pq_npts);
            uint32_t num_pq_chunks_32 = static_cast<uint32_t>(pq_ndims_val);
            pq_writer.write(reinterpret_cast<const char*>(&points_num_32),    sizeof(uint32_t));
            pq_writer.write(reinterpret_cast<const char*>(&num_pq_chunks_32), sizeof(uint32_t));
            pq_writer.write(reinterpret_cast<const char*>(reordered_pq.get()), total_pq_bytes);
            pq_writer.close();
            std::cout << "  Written: " << reorder_pq_out << std::endl;

            // Copy PQ pivots file
            std::string src_pivots = index_path_prefix + "_pq_pivots.bin";
            std::string dst_pivots = output_prefix + "_pq_pivots.bin";
            std::ifstream pivot_in(src_pivots, std::ios::binary);
            if (!pivot_in)
            {
                std::cerr << "  Warning: Cannot find PQ pivots: " << src_pivots << std::endl;
            }
            else
            {
                std::ofstream pivot_out(dst_pivots, std::ios::binary);
                pivot_out << pivot_in.rdbuf();
                pivot_in.close();
                pivot_out.close();
                std::cout << "  Copied pivots: " << dst_pivots << std::endl;
            }
        }
    }

    // ========================================
    // Step 9: Convert ground truth to new IDs
    // ========================================
    {
        std::cout << "\n=== Step 9: Convert ground truth to new IDs ===" << std::endl;

        uint32_t* gt_ids = nullptr;
        float* gt_dists = nullptr;
        size_t gt_nqueries, gt_k;
        diskann::load_truthset(orig_gt_file, gt_ids, gt_dists, gt_nqueries, gt_k);
        std::cout << "  Loaded GT: " << gt_nqueries << " queries, K=" << gt_k << std::endl;

        // Convert original IDs -> new IDs
        for (size_t i = 0; i < gt_nqueries * gt_k; i++)
        {
            uint32_t orig_id = gt_ids[i];
            if (orig_id >= original_to_new.size())
            {
                std::cerr << "  Error: GT ID " << orig_id << " out of range (max "
                          << original_to_new.size() - 1 << ")" << std::endl;
                delete[] gt_ids;
                delete[] gt_dists;
                return -1;
            }
            gt_ids[i] = original_to_new[orig_id];
        }

        // Write new GT: [int32 nqueries][int32 K][IDs uint32][dists float]
        std::string new_gt_file = output_prefix + "_gt.bin";
        std::ofstream gt_writer(new_gt_file, std::ios::binary);
        int32_t nq_i32 = static_cast<int32_t>(gt_nqueries);
        int32_t k_i32  = static_cast<int32_t>(gt_k);
        gt_writer.write(reinterpret_cast<char*>(&nq_i32),  sizeof(int32_t));
        gt_writer.write(reinterpret_cast<char*>(&k_i32),   sizeof(int32_t));
        gt_writer.write(reinterpret_cast<char*>(gt_ids),   gt_nqueries * gt_k * sizeof(uint32_t));
        gt_writer.write(reinterpret_cast<char*>(gt_dists), gt_nqueries * gt_k * sizeof(float));
        gt_writer.close();
        std::cout << "  Written: " << new_gt_file << std::endl;

        delete[] gt_ids;
        delete[] gt_dists;
    }

    // ========================================
    // Done
    // ========================================
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Complete! Total time: " << total_elapsed.count() << "s" << std::endl;
    std::cout << "Output index: " << new_index_file << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    std::string data_type, dist_fn;
    std::string index_path_prefix, data_bin;
    float sample_ratio;
    std::string orig_pq_compressed_file, orig_gt_file;
    bool skip_profiling = false;

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

        desc.add_options()("index_path_prefix",
                          po::value<std::string>(&index_path_prefix)->required(),
                          "Prefix path for existing LAANN index files");

        desc.add_options()("data_bin",
                          po::value<std::string>(&data_bin)->required(),
                          "Path to original data file (.bin format) for sampling queries");

        desc.add_options()("sample_ratio",
                          po::value<float>(&sample_ratio)->default_value(0.05f),
                          "Fraction of vectors to randomly sample as queries (e.g. 0.05 for 5%)");

        desc.add_options()("skip_profiling",
                          po::bool_switch(&skip_profiling)->default_value(false),
                          "Skip profiling step and use existing frequency file");

        desc.add_options()("orig_pq_compressed_file",
                          po::value<std::string>(&orig_pq_compressed_file)->required(),
                          "Path to original PQ compressed data file (DiskANN format)");

        desc.add_options()("orig_gt_file",
                          po::value<std::string>(&orig_gt_file)->required(),
                          "Path to ground truth file in original vector IDs");

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
        metric = diskann::Metric::L2;
    else if (dist_fn == "mips")
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == "cosine")
        metric = diskann::Metric::COSINE;
    else
    {
        std::cerr << "Error: Unsupported distance function. Use l2/mips/cosine" << std::endl;
        return -1;
    }

    try
    {
        if (data_type == "float")
        {
            return reorder_pages_by_frequency<float>(
                index_path_prefix, data_bin, sample_ratio,
                skip_profiling, metric,
                orig_pq_compressed_file, orig_gt_file);
        }
        else if (data_type == "uint8")
        {
            return reorder_pages_by_frequency<uint8_t>(
                index_path_prefix, data_bin, sample_ratio,
                skip_profiling, metric,
                orig_pq_compressed_file, orig_gt_file);
        }
        else if (data_type == "int8")
        {
            return reorder_pages_by_frequency<int8_t>(
                index_path_prefix, data_bin, sample_ratio,
                skip_profiling, metric,
                orig_pq_compressed_file, orig_gt_file);
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
