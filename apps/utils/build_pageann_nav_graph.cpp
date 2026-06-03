// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Navigation Graph Construction
// Samples vectors inline from the PageANN disk index — no separate
// generate_sample_for_nav_graph step or intermediate metadata files needed.
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "defaults.h"
#include "index.h"
#include "index_factory.h"
#include "parameters.h"
#include "utils.h"

namespace po = boost::program_options;

// Generate K evenly distributed page IDs from N total pages.
// Formula: selected_page[i] = i * (N-1) / (K-1)   (integer division)
// Example: K=4, N=10 → [0, 3, 6, 9]
// Matches the same formula used in build_laann_nav_graph.cpp and pq_flash_index.cpp.
std::vector<uint32_t> generate_sampled_page_ids(uint32_t num_total_pages, uint32_t num_sampled_pages) {
    std::vector<uint32_t> selected;
    if (num_total_pages == 0 || num_sampled_pages == 0) return selected;
    num_sampled_pages = std::min(num_sampled_pages, num_total_pages);
    selected.reserve(num_sampled_pages);
    if (num_sampled_pages == 1) { selected.push_back(0); return selected; }
    for (uint32_t i = 0; i < num_sampled_pages; ++i)
        selected.push_back((uint64_t)i * (num_total_pages - 1) / (num_sampled_pages - 1));
    return selected;
}

// Generate sampling indices within a page using priority order:
// top (0), last (capacity-1), second-to-last (capacity-2), ...
// Matches the same function used in build_laann_nav_graph.cpp.
std::vector<uint32_t> generate_sample_indices(uint32_t page_capacity, uint32_t samples_per_page) {
    std::vector<uint32_t> indices;
    indices.reserve(samples_per_page);
    if (samples_per_page == 0 || page_capacity == 0) return indices;
    indices.push_back(0);  // top vector first
    for (uint32_t i = 1; i < page_capacity && indices.size() < samples_per_page; ++i) {
        uint32_t idx = page_capacity - i;  // capacity-1, capacity-2, ...
        if (idx != 0) indices.push_back(idx);
    }
    return indices;
}

template<typename T>
int build_nav_graph(const std::string& index_file,
                    const std::string& output_prefix,
                    const std::string& dist_fn,
                    const std::string& data_type,
                    uint32_t samples_per_page,
                    uint32_t R, uint32_t L,
                    float alpha, uint32_t num_threads,
                    uint32_t num_sampled_pages = 0)
{
    std::cout << "========================================" << std::endl;
    std::cout << "Building PageANN Navigation Graph" << std::endl;
    std::cout << "========================================" << std::endl;

    // ── 1. Read PageANN index metadata ───────────────────────────────────────
    std::ifstream idx(index_file, std::ios::binary);
    if (!idx.is_open()) {
        std::cerr << "Error: Cannot open index file: " << index_file << std::endl;
        return -1;
    }

    uint32_t nr, nc;
    idx.read((char*)&nr, sizeof(uint32_t));
    idx.read((char*)&nc, sizeof(uint32_t));

    uint64_t disk_nnodes, disk_ndims, page_size_meta;
    uint64_t medoid_meta, max_node_len_meta, nnodes_per_sector_meta, pq_ndims_meta, max_degree_meta;
    idx.read((char*)&disk_nnodes,            sizeof(uint64_t));
    idx.read((char*)&disk_ndims,             sizeof(uint64_t));
    idx.read((char*)&page_size_meta,         sizeof(uint64_t));
    idx.read((char*)&medoid_meta,            sizeof(uint64_t));
    idx.read((char*)&max_node_len_meta,      sizeof(uint64_t));
    idx.read((char*)&nnodes_per_sector_meta, sizeof(uint64_t));
    idx.read((char*)&pq_ndims_meta,          sizeof(uint64_t));
    idx.read((char*)&max_degree_meta,        sizeof(uint64_t));

    const size_t   points_num      = static_cast<size_t>(disk_nnodes);
    const size_t   dim             = static_cast<size_t>(disk_ndims);
    const uint32_t nnodes_per_page = static_cast<uint32_t>(nnodes_per_sector_meta);

    std::cout << "Index: " << index_file << std::endl;
    std::cout << "  Points: " << points_num << ", Dim: " << dim
              << ", Nodes/page: " << nnodes_per_page << std::endl;

    // ── 2. Determine sampling mode and compute selected pages ────────────────
    const uint32_t num_total_pages = static_cast<uint32_t>((points_num + nnodes_per_page - 1) / nnodes_per_page);
    bool page_subsampling = (num_sampled_pages > 0 && num_sampled_pages < num_total_pages);

    // Clamp samples_per_page to page capacity
    uint32_t actual_samples_per_page = std::min(samples_per_page, nnodes_per_page);
    if (actual_samples_per_page == 0) actual_samples_per_page = 1;

    uint32_t num_samples;
    std::vector<uint32_t> selected_pages;

    if (page_subsampling) {
        // Page subsampling: evenly distribute num_sampled_pages across all pages,
        // taking 1 vector (top) per selected page.
        selected_pages = generate_sampled_page_ids(num_total_pages, num_sampled_pages);
        num_samples = static_cast<uint32_t>(selected_pages.size());
        std::cout << "  Mode: page subsampling — " << num_samples << " pages out of " << num_total_pages
                  << " (every ~" << num_total_pages / num_samples << " pages)" << std::endl;
    } else {
        // Normal mode: sample actual_samples_per_page vectors from every page.
        // Last page may have fewer vectors.
        uint32_t last_page_vectors = static_cast<uint32_t>(points_num - (uint64_t)(num_total_pages - 1) * nnodes_per_page);
        uint32_t last_page_samples = std::min(actual_samples_per_page, last_page_vectors);
        num_samples = (num_total_pages - 1) * actual_samples_per_page + last_page_samples;
        std::cout << "  Mode: " << actual_samples_per_page << " sample(s)/page — "
                  << num_samples << " total samples from " << num_total_pages << " pages" << std::endl;
    }

    // Pre-compute sample indices within a full page and the last page
    const std::vector<uint32_t> page_sample_indices =
        generate_sample_indices(nnodes_per_page, actual_samples_per_page);
    const uint32_t last_page_vectors =
        static_cast<uint32_t>(points_num - (uint64_t)(num_total_pages - 1) * nnodes_per_page);
    const uint32_t last_page_samples = page_subsampling ? 1 : std::min(actual_samples_per_page, last_page_vectors);
    const std::vector<uint32_t> last_page_sample_indices =
        generate_sample_indices(last_page_vectors, last_page_samples);

    std::cout << "  Sampling order within page: ";
    for (size_t i = 0; i < page_sample_indices.size() && i < 8; ++i)
        std::cout << page_sample_indices[i] << " ";
    if (page_sample_indices.size() > 8) std::cout << "...";
    std::cout << std::endl;

    // ── 3. Extract sampled vectors from disk index ────────────────────────────
    T* sampled_data = new T[(size_t)num_samples * dim]();
    const uint32_t report_interval = std::max(1000U, num_samples / 100);
    uint32_t sample_count = 0;

    std::cout << "Extracting sampled vectors..." << std::endl;

    if (page_subsampling) {
        for (uint32_t sel = 0; sel < selected_pages.size(); ++sel) {
            const uint64_t file_offset = ((uint64_t)selected_pages[sel] + 1) * diskann::defaults::SECTOR_LEN;
            idx.seekg(file_offset, std::ios::beg);
            idx.read((char*)(sampled_data + (size_t)sample_count * dim), dim * sizeof(T));
            if (!idx.good()) {
                std::cerr << "\nError reading page " << selected_pages[sel] << std::endl;
                delete[] sampled_data; return -1;
            }
            sample_count++;
            if (sample_count % report_interval == 0 || sample_count == num_samples)
                std::cout << "\r  " << sample_count << "/" << num_samples << std::flush;
        }
    } else {
        for (uint32_t page_id = 0; page_id < num_total_pages; ++page_id) {
            const uint64_t page_offset = ((uint64_t)page_id + 1) * diskann::defaults::SECTOR_LEN;
            bool is_last = (page_id == num_total_pages - 1);
            const std::vector<uint32_t>& indices = is_last ? last_page_sample_indices : page_sample_indices;

            for (uint32_t vec_offset : indices) {
                const uint64_t file_offset = page_offset + (uint64_t)vec_offset * max_node_len_meta;
                idx.seekg(file_offset, std::ios::beg);
                idx.read((char*)(sampled_data + (size_t)sample_count * dim), dim * sizeof(T));
                if (!idx.good()) {
                    std::cerr << "\nError reading page " << page_id << " vec " << vec_offset << std::endl;
                    delete[] sampled_data; return -1;
                }
                sample_count++;
            }
            if (sample_count % report_interval == 0 || sample_count == num_samples)
                std::cout << "\r  " << sample_count << "/" << num_samples << std::flush;
        }
    }
    std::cout << std::endl;
    idx.close();

    // ── 4. Write temp sampled data file (required by DiskANN index.build) ────
    const std::string temp_data_file = output_prefix + "_nav_temp_sampled.bin";
    diskann::save_bin<T>(temp_data_file, sampled_data, num_samples, dim);
    delete[] sampled_data;

    // ── 5. Determine metric ───────────────────────────────────────────────────
    diskann::Metric metric;
    if      (dist_fn == "l2")     metric = diskann::Metric::L2;
    else if (dist_fn == "mips")   metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == "cosine") metric = diskann::Metric::COSINE;
    else {
        std::cerr << "Error: Unsupported dist_fn. Use l2/mips/cosine" << std::endl;
        std::remove(temp_data_file.c_str());
        return -1;
    }

    // ── 6. Build Vamana graph on sampled data ─────────────────────────────────
    diskann::cout << "Building Vamana graph: R=" << R << " L=" << L
                  << " alpha=" << alpha
                  << " threads=" << (num_threads == 0 ? omp_get_num_procs() : (int)num_threads)
                  << std::endl;

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                  .with_alpha(alpha)
                                  .with_saturate_graph(false)
                                  .with_num_threads(num_threads)
                                  .build();
    auto filter_params = diskann::IndexFilterParamsBuilder().build();
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(dim)
                      .with_max_points(num_samples)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(data_type)
                      .is_dynamic_index(false)
                      .with_index_write_params(index_build_params)
                      .is_enable_tags(false)
                      .is_pq_dist_build(false)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();

    auto build_start = std::chrono::high_resolution_clock::now();
    index->build(temp_data_file, num_samples, filter_params);
    double build_secs = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - build_start).count();
    diskann::cout << "Vamana build time: " << build_secs << "s" << std::endl;

    // ── 7. Save nav graph with embedded sampling metadata ─────────────────────
    const std::string temp_vamana_prefix = output_prefix + "_nav_temp";
    const std::string final_graph_file   = output_prefix + "_nav_graph.index";
    const std::string final_data_file    = output_prefix + "_nav_data.bin";

    index->save(temp_vamana_prefix.c_str());

    // Read Vamana graph
    std::ifstream temp_in(temp_vamana_prefix, std::ios::binary);
    if (!temp_in.is_open()) {
        std::cerr << "Error: Cannot open temp Vamana graph: " << temp_vamana_prefix << std::endl;
        return -1;
    }
    uint64_t file_size_orig;
    uint32_t max_degree_orig, entry_point_orig;
    uint64_t num_frozen_pts;
    temp_in.read((char*)&file_size_orig,   sizeof(uint64_t));
    temp_in.read((char*)&max_degree_orig,  sizeof(uint32_t));
    temp_in.read((char*)&entry_point_orig, sizeof(uint32_t));
    temp_in.read((char*)&num_frozen_pts,   sizeof(uint64_t));

    std::vector<std::vector<uint32_t>> adjacency_lists;
    size_t bytes_read = sizeof(uint64_t) + 2 * sizeof(uint32_t) + sizeof(uint64_t);
    while (bytes_read < file_size_orig) {
        uint32_t num_neighbors;
        temp_in.read((char*)&num_neighbors, sizeof(uint32_t));
        std::vector<uint32_t> neighbors(num_neighbors);
        temp_in.read((char*)neighbors.data(), num_neighbors * sizeof(uint32_t));
        adjacency_lists.push_back(std::move(neighbors));
        bytes_read += sizeof(uint32_t) + num_neighbors * sizeof(uint32_t);
    }
    temp_in.close();

    const uint32_t num_nodes = static_cast<uint32_t>(adjacency_lists.size());

    uint64_t final_file_size = sizeof(uint64_t) + 5 * sizeof(uint32_t);
    for (const auto& nbrs : adjacency_lists)
        final_file_size += sizeof(uint32_t) + nbrs.size() * sizeof(uint32_t);

    // Header: [file_size][num_nodes][max_degree][entry_point][samples_per_page][nnodes_per_page][num_total_pages]
    // samples_per_page (5th field): number of vectors sampled from each page in priority order
    //   (top vector first, then last, second-to-last, ...). In page subsampling mode this is 1.
    // num_total_pages (7th field): 0 = normal mode (all pages sampled);
    //   >0 = page subsampling mode — search uses generate_sampled_page_ids(num_total_pages, num_nodes)
    //   to reconstruct which pages were selected.
    const uint32_t header_samples_per_page = page_subsampling ? 1 : actual_samples_per_page;
    const uint32_t header_total_pages      = page_subsampling ? num_total_pages : 0;

    std::ofstream graph_out(final_graph_file, std::ios::binary);
    graph_out.write((char*)&final_file_size,        sizeof(uint64_t));
    graph_out.write((char*)&num_nodes,              sizeof(uint32_t));
    graph_out.write((char*)&max_degree_orig,        sizeof(uint32_t));
    graph_out.write((char*)&entry_point_orig,       sizeof(uint32_t));
    graph_out.write((char*)&header_samples_per_page, sizeof(uint32_t));
    graph_out.write((char*)&nnodes_per_page,        sizeof(uint32_t));
    graph_out.write((char*)&header_total_pages,     sizeof(uint32_t));
    for (const auto& nbrs : adjacency_lists) {
        uint32_t n = static_cast<uint32_t>(nbrs.size());
        if (n > 255) {
            std::cerr << "Warning: node has " << n << " neighbors, clamping to 255" << std::endl;
            n = 255;
        }
        graph_out.write((char*)&n, sizeof(uint32_t));
        graph_out.write((char*)nbrs.data(), n * sizeof(uint32_t));
    }
    graph_out.close();

    // Move nav data, remove temp files
    std::rename((temp_vamana_prefix + ".data").c_str(), final_data_file.c_str());
    std::remove(temp_vamana_prefix.c_str());
    std::remove(temp_data_file.c_str());

    std::cout << "========================================" << std::endl;
    std::cout << "Navigation Graph Built Successfully!" << std::endl;
    std::cout << "  Graph      : " << final_graph_file << std::endl;
    std::cout << "  Data       : " << final_data_file << std::endl;
    std::cout << "  Entry point: " << entry_point_orig << std::endl;
    std::cout << "  Max degree : " << max_degree_orig << std::endl;
    std::cout << "  Samples    : " << num_samples
              << " (" << (page_subsampling ? "1 per selected page" : std::to_string(actual_samples_per_page) + "/page") << ")" << std::endl;
    std::cout << "  Build time : " << build_secs << "s" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    std::string data_type, dist_fn, index_file, output_prefix;
    uint32_t R, L, num_threads, num_sampled_pages, samples_per_page;
    float alpha;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn",
                           po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("index_file",
                           po::value<std::string>(&index_file)->required(),
                           "PageANN disk index file (.index)");
        desc.add_options()("output_prefix",
                           po::value<std::string>(&output_prefix)->required(),
                           "output prefix for nav graph files");
        desc.add_options()("samples_per_page,S",
                           po::value<uint32_t>(&samples_per_page)->default_value(1),
                           "Number of vectors to sample from each page (default: 1). "
                           "Sampling order: top(0), last, second-to-last, ... "
                           "Ignored when --num_sampled_pages is set.");
        desc.add_options()("num_sampled_pages",
                           po::value<uint32_t>(&num_sampled_pages)->default_value(0),
                           "Sample exactly this many pages using evenly distributed selection: "
                           "selected_page[i] = i*(N-1)/(K-1). Takes 1 vector (top) per selected page. "
                           "0 = disabled (sample all pages using --samples_per_page). "
                           "Use when memory budget limits the total number of nav graph nodes.");
        desc.add_options()("max_degree,R",
                           po::value<uint32_t>(&R)->default_value(64),
                           "max graph degree");
        desc.add_options()("Lbuild,L",
                           po::value<uint32_t>(&L)->default_value(100),
                           "build complexity");
        desc.add_options()("alpha",
                           po::value<float>(&alpha)->default_value(1.2f),
                           "alpha for graph density");
        desc.add_options()("num_threads,T",
                           po::value<uint32_t>(&num_threads)->default_value(0),
                           "threads (0 = all available)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc; return 0; }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    try {
        if      (data_type == "float")  return build_nav_graph<float>  (index_file, output_prefix, dist_fn, data_type, samples_per_page, R, L, alpha, num_threads, num_sampled_pages);
        else if (data_type == "int8")   return build_nav_graph<int8_t> (index_file, output_prefix, dist_fn, data_type, samples_per_page, R, L, alpha, num_threads, num_sampled_pages);
        else if (data_type == "uint8")  return build_nav_graph<uint8_t>(index_file, output_prefix, dist_fn, data_type, samples_per_page, R, L, alpha, num_threads, num_sampled_pages);
        else {
            std::cerr << "Error: Unsupported data type. Use int8/uint8/float" << std::endl;
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
