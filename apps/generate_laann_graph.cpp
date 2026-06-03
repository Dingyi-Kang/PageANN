// LAANN: Dynamic Graph Construction Tool 
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

/**
 * @file generate_laann_graph.cpp
 * @brief CLI tool to generate a LAANN graph
 *
 * This tool transforms a traditional Vamana graph into a LAANN graph improve cache utilization.
 * Unlike PageANN which focuses on static page-level organization, LAANN uses dynamic
 * cache patterns and provides better CPU-IO scheduling.
 *
 * @usage ./generate_laann_graph --data_type <float|int8|uint8> --dist_fn <l2|mips|cosine>
 *        --data_path <base_data_file> --vamana_index_path_prefix <index_prefix>
 *        --min_degree_per_node <degree> --num_PQ_chunks <chunks> --R <vamana_max_degree>
 *        --L <search_list_size>
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
 * Main entry point for LAANN graph generation.
 * Parses command-line arguments, validates inputs, and invokes the LAANN-specific
 * build_laann_graph function with appropriate parameters.
 */
int main(int argc, char **argv)
{
    std::string index_prefix_path, data_type, dist_fn, base_data_file;
    uint32_t expected_degree_per_vector, num_pq_chunks_32, maxVamanaDegree, L, fill_L;
    bool use_greedy_grouping;
    std::string new_to_old_ids_map_file;

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
        desc.add_options()("min_degree_per_node,minND", po::value<uint32_t>(&expected_degree_per_vector)->required(),
            "Expected degree per vector in output LAANN graph");
        desc.add_options()("num_PQ_chunks", po::value<uint32_t>(&num_pq_chunks_32)->required(),
            "Number of Product Quantization chunks for vector compression");
        desc.add_options()("R", po::value<uint32_t>(&maxVamanaDegree)->required(),
            "Maximum degree of input Vamana index (used for memory estimation)");
        desc.add_options()("L", po::value<uint32_t>(&L)->required(),
            "Search list size for greedy grouping (must be >= megaNode_capacity)");
        desc.add_options()("fill_L", po::value<uint32_t>(&fill_L)->default_value(100),
            "Search list size for filling spare neighbor slots per page (beam search)");
        desc.add_options()("use_greedy_grouping",
            po::bool_switch(&use_greedy_grouping)->default_value(false),
            "Use greedy search-based grouping instead of BFS graph-topology merging (default: BFS)");
        desc.add_options()("new_to_old_ids_map_file", po::value<std::string>(&new_to_old_ids_map_file)->default_value(""),
            "Pre-computed new-to-old ID map file; if provided, skips the vector grouping step");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        // Validate that all required parameters were provided
        po::notify(vm);
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

    // Invoke build_laann_graph with appropriate template parameter
    // This calls the LAANN-specific implementation that uses:
    // - R as the first parameter (maxVamanaDegree)
    // - expected_degree_per_vector as the second parameter
    try
    {
        std::cout << "Generating LAANN graph ..." << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  Index prefix: " << index_prefix_path << std::endl;
        std::cout << "  Data file: " << base_data_file << std::endl;
        std::cout << "  Max Vamana degree (R): " << maxVamanaDegree << std::endl;
        std::cout << "  Grouping method: " << (use_greedy_grouping ? "greedy search (L=" + std::to_string(L) + ")" : "BFS graph-topology") << std::endl;
        std::cout << "  Fill search list size (fill_L): " << fill_L << std::endl;
        std::cout << "  Expected degree per vector: " << expected_degree_per_vector << std::endl;
        std::cout << "  PQ chunks: " << num_pq_chunks_32 << std::endl;

        if (data_type == std::string("float"))
             diskann::build_laann_graph<float>(index_prefix_path, base_data_file, maxVamanaDegree,
                                                L, fill_L, expected_degree_per_vector, num_pq_chunks_32,
                                                metric, use_greedy_grouping, new_to_old_ids_map_file);
        if (data_type == std::string("int8"))
             diskann::build_laann_graph<int8_t>(index_prefix_path, base_data_file, maxVamanaDegree,
                                                 L, fill_L, expected_degree_per_vector, num_pq_chunks_32,
                                                 metric, use_greedy_grouping, new_to_old_ids_map_file);
        if (data_type == std::string("uint8"))
             diskann::build_laann_graph<uint8_t>(index_prefix_path, base_data_file, maxVamanaDegree,
                                                  L, fill_L, expected_degree_per_vector, num_pq_chunks_32,
                                                  metric, use_greedy_grouping, new_to_old_ids_map_file);

        std::cout << "LAANN graph generation completed successfully!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "Error during LAANN graph generation: " << std::string(e.what()) << std::endl;
        diskann::cerr << "LAANN graph generation failed. Check parameters and input files." << std::endl;
        return -1;
    }
}
