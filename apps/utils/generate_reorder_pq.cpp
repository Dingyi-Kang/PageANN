// PageANN: PQ Generation and Reordering Utility
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.
//
// This utility can:
// 1. Generate new PQ data from raw vectors and reorder by ID mapping
// 2. Reorder existing PQ data using a new ID mapping (for page reordering)

#include "math_utils.h"
#include "pq.h"
#include "partition.h"
#include <chrono>
#include <boost/program_options.hpp>
#include "filter_utils.h"

namespace po = boost::program_options;

#define KMEANS_ITERS_FOR_PQ 15

template <typename T>
bool generate_reorder_pq(const std::string &data_path,
                         const std::string &pq_input_prefix,
                         const std::string &id_map_file,
                         const std::string &output_prefix,
                         const size_t num_pq_centers,
                         const size_t num_pq_chunks,
                         const bool gen_pq)
{
    const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 256000;

    std::string pq_pivots_path = pq_input_prefix + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = pq_input_prefix + "_pq_compressed.bin";
    std::string reorder_pq_compressed_vectors_path = output_prefix + "_reorder_pq_compressed.bin";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "========================================" << std::endl;
    std::cout << "PQ Generation and Reordering Utility" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  PQ input prefix: " << pq_input_prefix << std::endl;
    std::cout << "  ID mapping file: " << id_map_file << std::endl;
    std::cout << "  Output prefix: " << output_prefix << std::endl;
    std::cout << "  Generate new PQ: " << (gen_pq ? "yes" : "no") << std::endl;

    // Step 0: Generate new PQ data (optional)
    if (gen_pq)
    {
        std::cout << "\n=== Step 0: Generate PQ data ===" << std::endl;
        size_t points_num, dim;
        diskann::get_bin_metadata(data_path.c_str(), points_num, dim);
        size_t num_sample_points = points_num > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : points_num;
        float sampling_rate = (float)num_sample_points / points_num;

        size_t train_size, train_dim;
        float *train_data;
        gen_random_slice<T>(data_path, sampling_rate, train_data, train_size, train_dim);
        std::cout << "  Loaded sample data of size " << train_size << std::endl;

        std::cout << "  Generating PQ pivots..." << std::endl;
        diskann::generate_pq_pivots(train_data, train_size, (uint32_t)train_dim, (uint32_t)num_pq_centers,
                                    (uint32_t)num_pq_chunks, KMEANS_ITERS_FOR_PQ, pq_pivots_path);

        std::cout << "  Generating PQ compressed data..." << std::endl;
        diskann::generate_pq_data_from_pivots<T>(data_path, (uint32_t)num_pq_centers, (uint32_t)num_pq_chunks,
                                                 pq_pivots_path, pq_compressed_vectors_path, false);
        delete[] train_data;
        std::cout << "  PQ data generated." << std::endl;
    }

    // Step 1: Load PQ data into RAM
    std::cout << "\n=== Step 1: Load PQ data ===" << std::endl;
    int all_pq_npts, pq_ndims;
    std::ifstream pqReader;
    pqReader.exceptions(std::ios::badbit | std::ios::failbit);
    pqReader.open(pq_compressed_vectors_path, std::ios::binary);
    pqReader.read(reinterpret_cast<char*>(&all_pq_npts), sizeof(int));
    pqReader.read(reinterpret_cast<char*>(&pq_ndims), sizeof(int));

    std::cout << "  Number of points: " << all_pq_npts << std::endl;
    std::cout << "  PQ dimensions: " << pq_ndims << std::endl;

    size_t n_pts = static_cast<size_t>(all_pq_npts);
    size_t dims = static_cast<size_t>(pq_ndims);
    size_t total_bytes = n_pts * dims * sizeof(uint8_t);
    std::cout << "  Allocating " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    std::unique_ptr<uint8_t[]> pq_data;
    try
    {
        pq_data = std::make_unique<uint8_t[]>(total_bytes);
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return false;
    }
    pqReader.read(reinterpret_cast<char*>(pq_data.get()), total_bytes);
    pqReader.close();
    std::cout << "  Loaded PQ data." << std::endl;

    // Step 2: Load ID mapping and reorder
    std::cout << "\n=== Step 2: Load ID mapping ===" << std::endl;
    std::cout << "  Reading from: " << id_map_file << std::endl;
    std::vector<uint32_t> new_to_original = diskann::loadTags(id_map_file, "");
    std::cout << "  Loaded " << new_to_original.size() << " ID mappings" << std::endl;

    // Step 3: Reorder PQ data
    // new_to_original[new_id] = original_id
    // reordered_pq[new_id] = pq_data[original_id]
    std::cout << "\n=== Step 3: Reorder PQ data ===" << std::endl;
    const size_t pq_size = dims * sizeof(uint8_t);
    std::unique_ptr<uint8_t[]> reordered_pq = std::make_unique<uint8_t[]>(total_bytes);

    for (size_t new_id = 0; new_id < new_to_original.size(); new_id++)
    {
        uint32_t original_id = new_to_original[new_id];
        memcpy(&reordered_pq[new_id * dims], &pq_data[original_id * dims], pq_size);
    }
    std::cout << "  Reordered " << new_to_original.size() << " PQ vectors" << std::endl;

    // Step 4: Write reordered PQ data
    std::cout << "\n=== Step 4: Write output ===" << std::endl;
    std::ofstream outFile(reorder_pq_compressed_vectors_path, std::ios::binary);
    if (!outFile)
    {
        std::cerr << "Error: Unable to open file " << reorder_pq_compressed_vectors_path << " for writing." << std::endl;
        return false;
    }

    // LAANN PQ format: [uint32_t npts][uint32_t num_chunks][data]
    uint32_t points_num_32   = static_cast<uint32_t>(all_pq_npts);
    uint32_t num_pq_chunks_32 = static_cast<uint32_t>(pq_ndims);

    outFile.write(reinterpret_cast<const char*>(&points_num_32),    sizeof(uint32_t));
    outFile.write(reinterpret_cast<const char*>(&num_pq_chunks_32), sizeof(uint32_t));
    outFile.write(reinterpret_cast<const char*>(reordered_pq.get()), total_bytes);
    outFile.close();

    std::cout << "  Written: " << reorder_pq_compressed_vectors_path << std::endl;

    // Step 5: Copy pivot file to output prefix
    std::cout << "\n=== Step 5: Copy pivot file ===" << std::endl;
    std::string output_pivots_path = output_prefix + "_pq_pivots.bin";
    std::ifstream pivotIn(pq_pivots_path, std::ios::binary);
    if (!pivotIn)
    {
        std::cerr << "Warning: Cannot open pivot file: " << pq_pivots_path << std::endl;
    }
    else
    {
        std::ofstream pivotOut(output_pivots_path, std::ios::binary);
        pivotOut << pivotIn.rdbuf();
        pivotIn.close();
        pivotOut.close();
        std::cout << "  Copied: " << pq_pivots_path << " -> " << output_pivots_path << std::endl;
    }

    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Complete! Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;

    return true;
}

int main(int argc, char **argv)
{
    std::string data_type, data_path, pq_input_prefix, id_map_file, output_prefix;
    uint32_t num_pq_chunks;
    bool gen_pq = false;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print help message");

        desc.add_options()("data_type",
                          po::value<std::string>(&data_type)->required(),
                          "Data type <float/uint8/int8>");

        desc.add_options()("data_path",
                          po::value<std::string>(&data_path)->default_value(""),
                          "Path to raw data file (required if --gen_pq is set)");

        desc.add_options()("pq_input_prefix",
                          po::value<std::string>(&pq_input_prefix)->required(),
                          "Prefix path for input PQ files (_pq_pivots.bin, _pq_compressed.bin)");

        desc.add_options()("id_map_file",
                          po::value<std::string>(&id_map_file)->required(),
                          "Path to new-to-original ID mapping file");

        desc.add_options()("output_prefix",
                          po::value<std::string>(&output_prefix)->required(),
                          "Prefix path for output reordered PQ file");

        desc.add_options()("num_pq_chunks",
                          po::value<uint32_t>(&num_pq_chunks)->default_value(0),
                          "Number of PQ chunks (required if --gen_pq is set)");

        desc.add_options()("gen_pq",
                          po::bool_switch(&gen_pq)->default_value(false),
                          "Generate new PQ data before reordering");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);

        // Validate: if gen_pq is set, data_path and num_pq_chunks are required
        if (gen_pq && (data_path.empty() || num_pq_chunks == 0))
        {
            std::cerr << "Error: --data_path and --num_pq_chunks are required when --gen_pq is set" << std::endl;
            return -1;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    const size_t num_pq_centers = 256;

    if (data_type == "float")
        return generate_reorder_pq<float>(data_path, pq_input_prefix, id_map_file, output_prefix,
                                          num_pq_centers, num_pq_chunks, gen_pq) ? 0 : -1;
    else if (data_type == "uint8")
        return generate_reorder_pq<uint8_t>(data_path, pq_input_prefix, id_map_file, output_prefix,
                                            num_pq_centers, num_pq_chunks, gen_pq) ? 0 : -1;
    else if (data_type == "int8")
        return generate_reorder_pq<int8_t>(data_path, pq_input_prefix, id_map_file, output_prefix,
                                           num_pq_centers, num_pq_chunks, gen_pq) ? 0 : -1;
    else
    {
        std::cerr << "Error: Unsupported data type. Use float/uint8/int8" << std::endl;
        return -1;
    }
}
