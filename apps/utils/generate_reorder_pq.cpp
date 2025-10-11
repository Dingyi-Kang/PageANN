// PageANN: Page-level Graph Index Generation
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include "math_utils.h"
#include "pq.h"
#include "partition.h"
#include <chrono>
#include "filter_utils.h"

#define KMEANS_ITERS_FOR_PQ 15

template <typename T>
bool generate_pq(const std::string &data_path, const std::string &index_prefix_path, const size_t num_pq_centers,
                 const size_t num_pq_chunks)
{
    const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 256000;//diskann value: 256000
    std::string pq_pivots_path = index_prefix_path + "_PQ" + std::to_string(num_pq_chunks) + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_PQ" + std::to_string(num_pq_chunks) + "_reorder_pq_compressed_tem.bin";
    std::string reorder_pq_compressed_vectors_path = index_prefix_path + "_PQ" + std::to_string(num_pq_chunks) + "_reorder_pq_compressed.bin";
    
    //step 0: generate new pq data
    auto start_time = std::chrono::high_resolution_clock::now();
    bool gen_pq = false;
    if(gen_pq){
        size_t points_num, dim;
        diskann::get_bin_metadata(data_path.c_str(), points_num, dim);
        size_t num_sample_points = points_num > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : points_num;
        float sampling_rate = (float)num_sample_points / points_num;
        // generates random sample and sets it to train_data and updates train_size
        size_t train_size, train_dim;
        float *train_data;
        gen_random_slice<T>(data_path, sampling_rate, train_data, train_size, train_dim);
        std::cout << "For computing pivots, loaded sample data of size " << train_size << std::endl;
        std::cout << "Generating pq pivots..." << std::endl;
        diskann::generate_pq_pivots(train_data, train_size, (uint32_t)train_dim, (uint32_t)num_pq_centers,
                                        (uint32_t)num_pq_chunks, KMEANS_ITERS_FOR_PQ, pq_pivots_path);
        std::cout << "Generating pq data from pivots..." << std::endl;
        diskann::generate_pq_data_from_pivots<T>(data_path, (uint32_t)num_pq_centers, (uint32_t)num_pq_chunks,
                                                pq_pivots_path, pq_compressed_vectors_path, false);
        delete[] train_data;
    }
    //step 1: load pq data into RAM
    int all_pq_npts, pq_ndims;
    std::ifstream pqAllPointsReader;
    pqAllPointsReader.exceptions(std::ios::badbit | std::ios::failbit);
    pqAllPointsReader.open(pq_compressed_vectors_path, std::ios::binary);
    pqAllPointsReader.seekg(0, pqAllPointsReader.beg);
    pqAllPointsReader.read((char *)&all_pq_npts, sizeof(int));
    pqAllPointsReader.read((char *)&pq_ndims, sizeof(int));//i.e., num_pq_chunks
    diskann::cout << "Number of points of compressed PQ data: " << all_pq_npts << std::endl;
    diskann::cout << "Number of dimensions of compressed PQ data: " << pq_ndims << std::endl;
    //reading all the data into an array
    size_t n_pts = static_cast<size_t>(all_pq_npts);
    size_t dims = static_cast<size_t>(pq_ndims);
    size_t total_bytes = n_pts * dims * sizeof(uint8_t);
    std::cout << "Allocating " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    std::unique_ptr<uint8_t[]> pq_compressed_page_all_points;
    try {
        pq_compressed_page_all_points = std::make_unique<uint8_t[]>(total_bytes);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    pqAllPointsReader.read((char*)(pq_compressed_page_all_points.get()), total_bytes);
    pqAllPointsReader.close();
    diskann::cout << "Finished reading pq data into buffer." << std::endl;
    //step 2: reorder the pq data; reorder the pq data to make its index corresponds to the id of the node
    //step 2.1 load the oldID to newID map
    std::string tags_file = index_prefix_path + "_original_to_new_ids_map.bin";
    diskann::cout << "Reading old id to new id map from: " << tags_file << std::endl;
    std::vector<uint32_t> location_to_tag = diskann::loadTags(tags_file, data_path);
    std::cout << "Size of ids map: " << location_to_tag.size() << std::endl;

    const size_t pq_size = dims * sizeof(uint8_t);
    std::unique_ptr<uint8_t[]> reorder_pq_data_buff = std::make_unique<uint8_t[]>(total_bytes);
    for (size_t old_node_ID = 0; old_node_ID < n_pts; old_node_ID++){
        uint32_t new_reordered_node_ID = location_to_tag[old_node_ID];
        memcpy(&reorder_pq_data_buff[new_reordered_node_ID * dims], &pq_compressed_page_all_points[old_node_ID * dims], pq_size);
    }

    //step 3: write new data
    std::ofstream outFile(reorder_pq_compressed_vectors_path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file "<< reorder_pq_compressed_vectors_path << " for writing.\n";
    }
    //write metadata: number of total points, number of cached points, num_pq_chunks_32, useID_pqIdx_map, useID_pqIdx_array
    uint32_t points_num_32 = static_cast<uint32_t>(all_pq_npts);
    uint32_t num_cached_nodes = points_num_32;
    uint32_t num_pq_chunks_32 = static_cast<uint32_t>(pq_ndims);
    outFile.write(reinterpret_cast<const char*>(&points_num_32), sizeof(points_num_32));
    outFile.write(reinterpret_cast<const char*>(&num_cached_nodes), sizeof(num_cached_nodes));
    outFile.write(reinterpret_cast<const char*>(&num_pq_chunks_32), sizeof(num_pq_chunks_32));
    //assume the memory is large enough; so no need of hash table or array for mapping; 
    uint8_t useID_pqIdx_map_u8 = 0;
    uint8_t useID_pqIdx_array_u8 = 0;
    outFile.write(reinterpret_cast<const char*>(&useID_pqIdx_map_u8), sizeof(useID_pqIdx_map_u8));
    outFile.write(reinterpret_cast<const char*>(&useID_pqIdx_array_u8), sizeof(useID_pqIdx_array_u8));
    outFile.write(reinterpret_cast<const char*>(reorder_pq_data_buff.get()), total_bytes);
    outFile.close();

    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Generated new PQ data using " << duration.count() << " seconds\n" << std::endl;
    std::remove(pq_compressed_vectors_path.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: \n"
                  << argv[0]
                  << "  <data_type[float/uint8/int8]>   <data_file[.bin]>"
                     "  <PQ_prefix_path>  <target-bytes/data-point>"
                  << std::endl;
    }
    else
    {
        const std::string data_path(argv[2]);
        const std::string index_prefix_path(argv[3]);
        const size_t num_pq_centers = 256;
        const size_t num_pq_chunks = (size_t)atoi(argv[4]);
        
        if (std::string(argv[1]) == std::string("float"))
            generate_pq<float>(data_path, index_prefix_path, num_pq_centers, num_pq_chunks);
        else if (std::string(argv[1]) == std::string("int8"))
            generate_pq<int8_t>(data_path, index_prefix_path, num_pq_centers, num_pq_chunks);
        else if (std::string(argv[1]) == std::string("uint8"))
            generate_pq<uint8_t>(data_path, index_prefix_path, num_pq_centers, num_pq_chunks);
        else
            std::cout << "Error. wrong file type" << std::endl;
    }
}
