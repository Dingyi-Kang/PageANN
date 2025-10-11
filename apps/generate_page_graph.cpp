// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Page Graph Construction Tool
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

/**
 * @file generate_page_graph.cpp
 * @brief CLI tool to generate a page-level graph index from a Vamana vector-level index.
 *
 * This tool transforms a traditional Vamana graph (where each node represents a single vector)
 * into a PageANN page-level graph (where multiple vectors are merged into pages for efficient
 * disk I/O). The resulting page graph enables better SSD utilization by reducing random access
 * patterns during approximate nearest neighbor search.
 *
 * @usage ./generate_page_graph --data_type <float|int8|uint8> --dist_fn <l2|mips|cosine>
 *        --data_path <base_data_file> --vamana_index_path_prefix <index_prefix>
 *        --min_degree_per_node <degree> --num_PQ_chunks <chunks> --R <vamana_max_degree>
 *        --mem_budget_in_GB <memory> --full_ooc <true|false>
 */

#include "common_includes.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include <string>
#include <boost/program_options.hpp>
#include "utils.h"
#include "disk_utils.h"
#include "defaults.h"
#include "program_options_utils.hpp"


namespace po = boost::program_options;

/**
 * Main entry point for page graph generation.
 * Parses command-line arguments, validates inputs, and invokes the appropriate
 * template instantiation of build_page_graph based on data type.
 */
int main(int argc, char **argv)
{
    std::string index_prefix_path, data_type, dist_fn, base_data_file, full_ooc_str;
    uint32_t min_degree_per_node, num_pq_chunks_32, maxVamanaDegree;
    bool full_ooc = false;
    float memBudgetInGB = 0.0f;

    // Parse command-line arguments using Boost Program Options
    try
    {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
            "Data type of vectors: float, int8, or uint8");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
            "Distance function: l2 (Euclidean), mips (maximum inner product), or cosine");
        desc.add_options()("data_path", po::value<std::string>(&base_data_file)->required(),
            "Path to base dataset in binary format");
        desc.add_options()("vamana_index_path_prefix", po::value<std::string>(&index_prefix_path)->required(),
            "Path prefix for input Vamana graph index files");
        desc.add_options()("min_degree_per_node,minND", po::value<uint32_t>(&min_degree_per_node)->required(),
            "Target minimum degree per node in output page graph");
        desc.add_options()("num_PQ_chunks", po::value<uint32_t>(&num_pq_chunks_32)->required(),
            "Number of Product Quantization chunks for vector compression");
        desc.add_options()("R", po::value<uint32_t>(&maxVamanaDegree)->required(),
            "Maximum degree of input Vamana index (used for memory estimation)");
        desc.add_options()("mem_budget_in_GB", po::value<float>(&memBudgetInGB)->required(),
            "Memory budget in GB for the searching process (affects PQ caching strategy)");
        desc.add_options()("full_ooc", po::value<std::string>(&full_ooc_str)->required(),
            "Enable fully out-of-core mode (true/false). When true, minimizes in-memory data structures");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        // Validate that all required parameters were provided
        po::notify(vm);

        // Convert string boolean parameter to actual bool type
        if (full_ooc_str == "true" || full_ooc_str == "True")
            full_ooc = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Validate data type is one of the supported types
    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
    {
        std::cout << "Error: Unsupported data type '" << data_type
                  << "'. Supported types: float, int8, uint8" << std::endl;
        return -1;
    }

    // Parse and validate distance metric
    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cout << "Error: Unsupported distance function '" << dist_fn
                  << "'. Supported functions: l2, mips, cosine" << std::endl;
        return -1;
    }

    // Invoke build_page_graph with appropriate template parameter based on data type
    // Each data type (float, int8_t, uint8_t) requires separate template instantiation
    try
    {
        if (data_type == std::string("float"))
             diskann::build_page_graph<float>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, full_ooc);
        if (data_type == std::string("int8"))
             diskann::build_page_graph<int8_t>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, full_ooc);
        if (data_type == std::string("uint8"))
             diskann::build_page_graph<uint8_t>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, full_ooc);
    }
    catch (const std::exception &e)
    {
        std::cout << "Error during page graph generation: " << std::string(e.what()) << std::endl;
        diskann::cerr << "Page graph generation failed. Check parameters and input files." << std::endl;
        return -1;
    }
}
