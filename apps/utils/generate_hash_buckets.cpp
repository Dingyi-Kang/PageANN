// PageANN: Page-level Graph Index Generation
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <numeric>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"
#include "defaults.h"
#include "index_factory.h"

namespace po = boost::program_options;

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))

template <typename T>
void aux_main(const std::string &index_filepath, const size_t min_num_per_bucket, bool compute_from_beginning, const float sample_ratio, const std::string data_file_to_use)
{
    std::cout << "min_num_per_bucket: " << min_num_per_bucket << std::endl;
    if (compute_from_beginning){
        std::string tags_file = index_filepath + "_new_to_old_ids_map.bin";
        diskann::cout << "Reading new id to old id map from: " << tags_file << std::endl;
        std::vector<uint32_t> newID_to_oldID_map = diskann::loadTags(tags_file, data_file_to_use);
        std::cout << "Size of ids map: " << newID_to_oldID_map.size() << std::endl;

        size_t points_num, dim;
        diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);
        std::shared_ptr<diskann::InMemOOCDataStore<T>> data_store = diskann::IndexFactory::construct_ooc_datastore<T>(diskann::DataStoreStrategy::MEMORY, points_num, dim, diskann::Metric::L2);
        data_store->load(data_file_to_use);
        diskann::cout << "Finish loading data into data store." << std::endl;

        size_t approx_sampled_nodes = static_cast<size_t>(points_num * sample_ratio);
        std::cout << "Total nodes in dataset: " << points_num << std::endl;
        std::cout << "Sample ratio: " << sample_ratio << std::endl;
        std::cout << "Approximate number of nodes to be sampled: " << approx_sampled_nodes << std::endl;

        std::string projection_matrix_file = index_filepath + "_projection_matrix_file.bin";

        //get a projection matrix
        const int numProjections = 32;
        const int dimensions = static_cast<int>(dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> distribution(0.0, 1.0);
        std::vector<std::vector<float>> projectionMatrix(numProjections, std::vector<float>(dimensions));
        for (int i = 0; i < numProjections; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                projectionMatrix[i][j] = distribution(gen);
            }
        }

        // Write the projection matrix to a file
        std::ofstream outFilePM(projection_matrix_file, std::ios::binary);
        if (!outFilePM) {
            std::cerr << "Error: Unable to open file for writing.\n";
            return;
        }
        outFilePM.write(reinterpret_cast<const char*>(&numProjections), sizeof(numProjections));
        outFilePM.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));

        for (int i = 0; i < numProjections; ++i) {
            outFilePM.write(reinterpret_cast<const char*>(projectionMatrix[i].data()), dimensions * sizeof(float));
        }
        outFilePM.close();
        std::cout << "Projection matrix written to file.\n";
        std::cout << "Num of projections: " << numProjections << ", number of dimensions: " << dimensions << std::endl;

        T *curr_vec_T = new T[data_store->get_aligned_dim()];
        std::unique_ptr<float[]> curr_vec_float = std::make_unique<float[]>(data_store->get_aligned_dim());
        std::vector<float> normalized(dimensions);
        tsl::robin_map<uint32_t, std::unique_ptr<std::vector<uint32_t>>> buckets;
        uint8_t bucketCapacity = std::numeric_limits<uint8_t>::max();

        size_t num_sampled = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t target_samples = approx_sampled_nodes;
        std::bernoulli_distribution sample_dist(sample_ratio);
        bool sample_all = (sample_ratio >= 1.0f);

        for(uint32_t new_node_id = 0; new_node_id < points_num; new_node_id++){
            if (!sample_all && !sample_dist(gen)) {
                continue;  // skip this point ~(1-sample_ratio) of the time
            }

            uint32_t old_node_id = newID_to_oldID_map[new_node_id];
            data_store->get_vector(old_node_id, curr_vec_T);
            diskann::convert_types<T, float>(curr_vec_T, curr_vec_float.get(), 1, dim);

            float norm = 0.0f;
            for (size_t j = 0; j < dimensions; ++j) {
                norm += curr_vec_float[j] * curr_vec_float[j];
            }
            norm = std::sqrt(norm);
            // Avoid division by zero (ensure numerical stability)
            float norm_inv = (norm > 1e-8f) ? (1.0f / norm) : 0.0f; //effectively zeros out the normalized vector when norm is too small to trust.
            for (size_t j = 0; j < dimensions; ++j) {
                normalized[j] = curr_vec_float[j] * norm_inv;
            }

            //projecting, hashing, and storing in buckets
            uint32_t hash = 0;
            for (int i = 0; i < numProjections; ++i) {
                float dotProduct = 0.0;
                for (size_t j = 0; j < dimensions; ++j) {
                    dotProduct += normalized[j] * projectionMatrix[i][j];
                }
                if (dotProduct >= 0) {
                    hash |= (1U << i); // Use (1ULL << i) to create a number where only the i-th bit is 1 //Perform a bitwise OR (|=) with hash to set the bit without affecting other bits
                }
            }

            // Assign to bucket
            auto iter = buckets.find(hash);
            if (iter != buckets.end()) {
                //uint8_t& count = bucket_counts[hash];
                if(iter->second->size() < bucketCapacity) {  // Ensure count does not exceed 255
                    iter->second->push_back(new_node_id);
                    ++num_sampled;
                }
            }
            else {
                auto new_bucket = std::make_unique<std::vector<uint32_t>>();
                new_bucket->reserve(255); // Optional: Preallocate space for efficiency
                new_bucket->push_back(new_node_id);
                buckets[hash] = std::move(new_bucket);
                //bucket_counts[hash] = 1;
                ++num_sampled;
            }

            // Progress reporting
            if (new_node_id % 10000 == 0 && new_node_id != 0) {
                float percent = 100.0f * new_node_id / points_num;
                float percent_samples = 100.0f * num_sampled / target_samples;
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                std::cout << "\rProgress: "
                        << std::setw(6) << std::fixed << std::setprecision(2) << percent << "% (" << new_node_id << "/" << points_num << ")"
                        << ", sampled: " << num_sampled << "/" << target_samples
                        << " (" << std::setprecision(2) << percent_samples << "% of target)"
                        << ", time used: " << static_cast<int>(elapsed) << "s"
                        << std::flush;
            }
        }
        delete[] curr_vec_T;

        // Write the new buckets into disk
        ///MARK: note this is not subset
        std::string buckets_file = index_filepath + "_buckets.bin";
        std::ofstream outFileBuckets(buckets_file, std::ios::binary);
        if (!outFileBuckets) {
            throw std::runtime_error("Failed to open file for writing.");
        }

        size_t new_bucketCount = buckets.size();
        size_t new_sampled_nodes = 0;
        for (const auto& [_, idVec] : buckets) {
            new_sampled_nodes += idVec->size();
        }

        outFileBuckets.write(reinterpret_cast<const char*>(&new_bucketCount), sizeof(new_bucketCount));
        outFileBuckets.write(reinterpret_cast<const char*>(&numProjections), sizeof(numProjections));
        outFileBuckets.write(reinterpret_cast<const char*>(&new_sampled_nodes), sizeof(new_sampled_nodes));

        for (const auto& [hash, idVec] : buckets) {
            size_t idCount = idVec->size();
            if (idCount > bucketCapacity) {
                std::cerr << "Error: idCount too large (" << idCount << "), must be ≤ 255." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            outFileBuckets.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
            uint8_t idCount_u8 = static_cast<uint8_t>(idCount);
            outFileBuckets.write(reinterpret_cast<const char*>(&idCount_u8), sizeof(uint8_t));
            outFileBuckets.write(reinterpret_cast<const char*>(idVec->data()), idCount * sizeof(uint32_t));
        }

        outFileBuckets.close();
        std::cout << "Buckets are saved to " << buckets_file << std::endl;
        std::cout << "Stats:" << std::endl;
        std::cout << "  Total non-empty buckets: " << new_bucketCount << std::endl;
        std::cout << "  Total nodes sampled: " << num_sampled << std::endl;
        std::cout << "  Total nodes remaining: " << new_sampled_nodes << std::endl;
        if (new_bucketCount > 0) {
            std::cout << "  Average bucket size: " << static_cast<double>(new_sampled_nodes) / new_bucketCount << std::endl;
        } else {
            std::cout << "  Warning: All buckets are empty." << std::endl;
        }
        return;
    }

    //if it is not compute_from_beginning, we reduce the sample vectors in the hash buckets to reduce the memory overhead
    std::string pq_compressed_reorder_file = std::string(index_filepath) + "_reorder_pq_compressed.bin";
    std::ifstream inPQFile(pq_compressed_reorder_file, std::ios::binary);
    if (!inPQFile.is_open()) {
        throw std::runtime_error("Failed to open reorder compressed PQ data file for reading.");
    }
    uint32_t points_num, num_pq_cached_nodes;
    inPQFile.read(reinterpret_cast<char*>(&points_num), sizeof(points_num));
    inPQFile.read(reinterpret_cast<char*>(&num_pq_cached_nodes), sizeof(num_pq_cached_nodes));
    inPQFile.close();
    if (points_num > num_pq_cached_nodes){
        diskann::cout << "Error: to customize the number of nodes to be sampled in hash buckets, all PQ data is needed to be loaded in memory." << std::endl;
        return;
    }

    std::string bucketsFile = std::string(index_filepath) + "_buckets.bin";
    std::ifstream inFile(bucketsFile, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Failed to open hash buckets file for reading.");
    }
    // Read the number of buckets
    size_t bucketCount, num_sampled_nodes_in_hash;
    int numProjections;
    inFile.read(reinterpret_cast<char*>(&bucketCount), sizeof(bucketCount));
    inFile.read(reinterpret_cast<char*>(&numProjections), sizeof(numProjections));
    inFile.read(reinterpret_cast<char*>(&num_sampled_nodes_in_hash), sizeof(num_sampled_nodes_in_hash));

    diskann::cout << "Loading the base hash buckets: " << std::endl;
    diskann::cout << "Number of buckets: " << bucketCount << std::endl;
    diskann::cout << "Number of hashed sampled nodes in base buckets: " << num_sampled_nodes_in_hash << std::endl;

    // Calculate target_num_sampled_nodes from sample_ratio
    size_t target_num_sampled_nodes = static_cast<size_t>(num_sampled_nodes_in_hash * sample_ratio);
    std::cout << "Sample ratio: " << sample_ratio << std::endl;
    std::cout << "Target number of sampled nodes: " << target_num_sampled_nodes << std::endl;

    if(target_num_sampled_nodes >=  num_sampled_nodes_in_hash){
        std::cout << "Error: the base hash buckets table has only " << num_sampled_nodes_in_hash << " sampled vectors. Cannot sample more than that." << std::endl;
        inFile.close();
        return;
    }

    tsl::robin_map<uint32_t, std::unique_ptr<std::vector<uint32_t>>> buckets;
    buckets.reserve(bucketCount);
//load all ids in hash buckets
    for (size_t i = 0; i < bucketCount; ++i) {
        uint32_t hash;
        uint8_t numIds;  // Number of node IDs in this bucket
        // Read the hash key
        inFile.read(reinterpret_cast<char*>(&hash), sizeof(uint32_t));
        // Read the number of IDs in the bucket
        inFile.read(reinterpret_cast<char*>(&numIds), sizeof(uint8_t));

        auto ids = std::make_unique<std::vector<uint32_t>>(numIds);
        inFile.read(reinterpret_cast<char*>(ids->data()), static_cast<std::size_t>(numIds) * sizeof(uint32_t));
        buckets.emplace(hash, std::move(ids));
    }
    inFile.close();

//evenly remove nodes from each bucket in hash table
    size_t total_to_remove = num_sampled_nodes_in_hash - target_num_sampled_nodes;
    size_t removed = 0;
    //const size_t min_num_per_bucket = 4;
    // We'll loop until we remove the desired number of nodes
    size_t pass = 0;
    while (removed < total_to_remove) {
        // Identify eligible buckets (those with > 3 nodes)
        std::vector<std::vector<uint32_t>*> eligible;
        for (auto& [_, vec_ptr] : buckets) {
            if (vec_ptr->size() > min_num_per_bucket) {
                eligible.push_back(vec_ptr.get());  // get raw pointer from unique_ptr
            }
        }

        if (eligible.empty()) {
            std::cout << "No eligible buckets (with > "<< min_num_per_bucket <<" nodes) left to remove from. Removed "
                      << removed << " out of " << total_to_remove << std::endl;
            break;
        }

        // Recalculate avg to remove per bucket in this round
        size_t remaining_to_remove = total_to_remove - removed;
        size_t avg_remove = std::max<size_t>(1, remaining_to_remove / eligible.size());

        // Shuffle buckets for fairness
        std::shuffle(eligible.begin(), eligible.end(), std::mt19937{std::random_device{}()});

        for (auto* ids : eligible) {
            if (removed >= total_to_remove) break;

            size_t can_remove = ids->size() - min_num_per_bucket;
            size_t this_remove = std::min(avg_remove, can_remove);

            if (this_remove > 0) {
                std::shuffle(ids->begin(), ids->end(), std::mt19937{std::random_device{}()});
                ids->erase(ids->begin(), ids->begin() + this_remove);
                removed += this_remove;
            }
        }
        std::cout << "After pass "<< pass++ <<", " << removed <<" nodes have been removed\n";
    }

    size_t num_nodes_remain = num_sampled_nodes_in_hash - removed;
    //because we prioritize even the size of each bucket with size of at most 2
    if (min_num_per_bucket <= 2  & num_nodes_remain > 0){
            std::cout << "Enable to remove buckets now\n";
            while (removed < total_to_remove) {
                // Identify eligible buckets (those with > 3 nodes)
                std::vector<std::vector<uint32_t>*> eligible;
                for (auto& [_, vec_ptr] : buckets) {
                    if (vec_ptr->size() > 0) {
                        eligible.push_back(vec_ptr.get());  // get raw pointer from unique_ptr
                    }
                }

                if (eligible.empty()) {
                    std::cout << "No eligible buckets left to remove." << std::endl;
                    break;
                }

                // Recalculate avg to remove per bucket in this round
                size_t remaining_to_remove = total_to_remove - removed;
                size_t avg_remove = std::max<size_t>(1, remaining_to_remove / eligible.size());

                // Shuffle buckets for fairness
                std::shuffle(eligible.begin(), eligible.end(), std::mt19937{std::random_device{}()});

                for (auto* ids : eligible) {
                    if (removed >= total_to_remove) break;

                    size_t can_remove = ids->size();
                    size_t this_remove = std::min(avg_remove, can_remove);

                    if (this_remove > 0) {
                        std::shuffle(ids->begin(), ids->end(), std::mt19937{std::random_device{}()});
                        ids->erase(ids->begin(), ids->begin() + this_remove);
                        removed += this_remove;
                    }
                }
                std::cout << "After pass "<< pass++ <<", " << removed <<" nodes have been removed\n";
            }
    }

    num_nodes_remain = num_sampled_nodes_in_hash - removed;
    std::cout << "Finished bucket-aware reduction. Total removed: " << removed
              << ", target to remove: " << total_to_remove
              <<", num of sampled nodes remaining in hash buckets: " << num_nodes_remain
              << std::endl;
    std::cout << "Finished generating new hash buckets.\n";

    //write the new buckets into disk
    std::string buckets_file = index_filepath + "_subset_buckets.bin";
    std::ofstream outFileBuckets(buckets_file, std::ios::binary);
    if (!outFileBuckets) {
        throw std::runtime_error("Failed to open file for writing.");
    }
    size_t new_bucketCount = buckets.size();
    outFileBuckets.write(reinterpret_cast<const char*>(&new_bucketCount), sizeof(new_bucketCount));
    outFileBuckets.write(reinterpret_cast<const char*>(&numProjections), sizeof(numProjections));
    outFileBuckets.write(reinterpret_cast<const char*>(&num_nodes_remain), sizeof(num_nodes_remain));

    // Write each bucket (hash and its associated vector of IDs)
    uint8_t bucketCapacity = std::numeric_limits<uint8_t>::max();
    for (const auto& [hash, idVec] : buckets) {
        // Write the number of elements in the vector
        size_t idCount = idVec->size();
        // Sanity check to make sure we can cast to uint8_t safely
        if (idCount > bucketCapacity) {
            std::cerr << "Error: idCount too large (" << idCount << "), must be ≤ 255." << std::endl;
            // Handle this situation appropriately (e.g., split the bucket, skip, or exit)
            std::exit(EXIT_FAILURE);
        }

        if (idCount == 0)
            continue;

        outFileBuckets.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        uint8_t idCount_u8 = static_cast<uint8_t>(idCount);
        outFileBuckets.write(reinterpret_cast<const char*>(&idCount_u8), sizeof(uint8_t));
        // Write the vector ids within the bucket
        outFileBuckets.write(reinterpret_cast<const char*>(idVec->data()), idCount * sizeof(uint32_t));
    }

    outFileBuckets.close();
    std::cout << "New buckets are saved to " << buckets_file << std::endl;

    return;
}//end of aux_main

int main(int argc, char **argv)
{
    std::string graph_index_prefix, data_type, compute_from_beginning_str, data_file_to_use;
    size_t min_num_per_bucket;
    float sample_ratio;
    bool compute_from_beginning = false;
    try
    {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("graph_index_prefix", po::value<std::string>(&graph_index_prefix)->required(), "Graph index prefix for files to read");
        desc.add_options()("min_num_per_bucket", po::value<size_t>(&min_num_per_bucket)->default_value(4), "min_num_per_bucket.");
        desc.add_options()("sample_ratio", po::value<float>(&sample_ratio)->default_value(1.0f), "Sample ratio (0.0, 1.0] to determine number of nodes to sample.");
        desc.add_options()("compute_from_beginning", po::value<std::string>(&compute_from_beginning_str)->default_value(std::string("false")), "Compute all hash buckets from beginning instead of sampling from existing buckets.");
        desc.add_options()("data_file_to_use", po::value<std::string>(&data_file_to_use)->default_value(std::string("")),
                                       "base data file");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (compute_from_beginning_str == "true" || compute_from_beginning_str == "True")
            compute_from_beginning = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
    {
        std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        return -1;
    }

    // Validation: if compute_from_beginning is true, sample_ratio must be > 0
    if (compute_from_beginning && (sample_ratio <= 0.0f || sample_ratio > 1.0f))
    {
        std::cerr << "Error: sample_ratio must be in range (0.0, 1.0] when compute_from_beginning is true." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
            aux_main<float>(graph_index_prefix, min_num_per_bucket, compute_from_beginning, sample_ratio, data_file_to_use);
        if (data_type == std::string("int8"))
            aux_main<int8_t>(graph_index_prefix, min_num_per_bucket, compute_from_beginning, sample_ratio, data_file_to_use);
        if (data_type == std::string("uint8"))
            aux_main<uint8_t>(graph_index_prefix, min_num_per_bucket, compute_from_beginning, sample_ratio, data_file_to_use);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Generate hash buckets failed." << std::endl;
        return -1;
    }
}
