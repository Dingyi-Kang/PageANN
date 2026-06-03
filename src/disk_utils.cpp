// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Page Graph Construction and Disk Layout
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "logger.h"
#include "disk_utils.h"
#include "filter_utils.h"
#include "cached_io.h"
#include "index.h"
#include "mkl.h"
#include "omp.h"
#include "percentile_stats.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "index_factory.h"
#include <random>
#include <bitset>
#include <filesystem>

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))

namespace diskann
{

void add_new_file_to_single_index(std::string index_file, std::string new_file)
{
    std::unique_ptr<uint64_t[]> metadata;
    uint64_t nr, nc;
    diskann::load_bin<uint64_t>(index_file, metadata, nr, nc);
    if (nc != 1)
    {
        std::stringstream stream;
        stream << "Error, index file specified does not have correct metadata. " << std::endl;
        throw diskann::ANNException(stream.str(), -1);
    }
    size_t index_ending_offset = metadata[nr - 1];
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ofstream writer(index_file, read_blk_size);
    size_t check_file_size = get_file_size(index_file);
    if (check_file_size != index_ending_offset)
    {
        std::stringstream stream;
        stream << "Error, index file specified does not have correct metadata "
                  "(last entry must match the filesize). "
               << std::endl;
        throw diskann::ANNException(stream.str(), -1);
    }

    cached_ifstream reader(new_file, read_blk_size);
    size_t fsize = reader.get_file_size();
    if (fsize == 0)
    {
        std::stringstream stream;
        stream << "Error, new file specified is empty. Not appending. " << std::endl;
        throw diskann::ANNException(stream.str(), -1);
    }

    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    char *dump = new char[read_blk_size];
    for (uint64_t i = 0; i < num_blocks; i++)
    {
        size_t cur_block_size =
            read_blk_size > fsize - (i * read_blk_size) ? fsize - (i * read_blk_size) : read_blk_size;
        reader.read(dump, cur_block_size);
        writer.write(dump, cur_block_size);
    }
    //    reader.close();
    //    writer.close();

    delete[] dump;
    std::vector<uint64_t> new_meta;
    for (uint64_t i = 0; i < nr; i++)
        new_meta.push_back(metadata[i]);
    new_meta.push_back(metadata[nr - 1] + fsize);

    diskann::save_bin<uint64_t>(index_file, new_meta.data(), new_meta.size(), 1);
}

double get_memory_budget(double search_ram_budget)
{
    double final_index_ram_limit = search_ram_budget;
    if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB)
    { // slack for space used by cached
      // nodes
        final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
}

double get_memory_budget(const std::string &mem_budget_str)
{
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
}

size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim,
                               const std::vector<std::string> &param_list)
{
    size_t num_pq_chunks = (size_t)(std::floor)(uint64_t(final_index_ram_limit / (double)points_num));
    diskann::cout << "Calculated num_pq_chunks :" << num_pq_chunks << std::endl;
    if (param_list.size() >= 6)
    {
        float compress_ratio = (float)atof(param_list[5].c_str());
        if (compress_ratio > 0 && compress_ratio <= 1)
        {
            size_t chunks_by_cr = (size_t)(std::floor)(compress_ratio * dim);

            if (chunks_by_cr > 0 && chunks_by_cr < num_pq_chunks)
            {
                diskann::cout << "Compress ratio:" << compress_ratio << " new #pq_chunks:" << chunks_by_cr << std::endl;
                num_pq_chunks = chunks_by_cr;
            }
            else
            {
                diskann::cout << "Compress ratio: " << compress_ratio << " #new pq_chunks: " << chunks_by_cr
                              << " is either zero or greater than num_pq_chunks: " << num_pq_chunks
                              << ". num_pq_chunks is unchanged. " << std::endl;
            }
        }
        else
        {
            diskann::cerr << "Compression ratio: " << compress_ratio << " should be in (0,1]" << std::endl;
        }
    }

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    diskann::cout << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector."
                  << std::endl;
    return num_pq_chunks;
}

template <typename T> T *generateRandomWarmup(uint64_t warmup_num, uint64_t warmup_dim, uint64_t warmup_aligned_dim)
{
    T *warmup = nullptr;
    warmup_num = 100000;
    diskann::cout << "Generating random warmup file with dim " << warmup_dim << " and aligned dim "
                  << warmup_aligned_dim << std::flush;
    diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
    std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    for (uint32_t i = 0; i < warmup_num; i++)
    {
        for (uint32_t d = 0; d < warmup_dim; d++)
        {
            warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
        }
    }
    diskann::cout << "..done" << std::endl;
    return warmup;
}

#ifdef EXEC_ENV_OLS
template <typename T>
T *load_warmup(MemoryMappedFiles &files, const std::string &cache_warmup_file, uint64_t &warmup_num,
               uint64_t warmup_dim, uint64_t warmup_aligned_dim)
{
    T *warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;

    if (files.fileExists(cache_warmup_file))
    {
        diskann::load_aligned_bin<T>(files, cache_warmup_file, warmup, warmup_num, file_dim, file_aligned_dim);
        diskann::cout << "In the warmup file: " << cache_warmup_file << " File dim: " << file_dim
                      << " File aligned dim: " << file_aligned_dim << " Expected dim: " << warmup_dim
                      << " Expected aligned dim: " << warmup_aligned_dim << std::endl;

        if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim)
        {
            std::stringstream stream;
            stream << "Mismatched dimensions in sample file. file_dim = " << file_dim
                   << " file_aligned_dim: " << file_aligned_dim << " index_dim: " << warmup_dim
                   << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
            diskann::cerr << stream.str();
            throw diskann::ANNException(stream.str(), -1);
        }
    }
    else
    {
        warmup = generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
    }
    return warmup;
}
#endif

template <typename T>
T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num, uint64_t warmup_dim,
               uint64_t warmup_aligned_dim)
{
    T *warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;

    if (file_exists(cache_warmup_file))
    {
        diskann::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num, file_dim, file_aligned_dim);
        if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim)
        {
            std::stringstream stream;
            stream << "Mismatched dimensions in sample file. file_dim = " << file_dim
                   << " file_aligned_dim: " << file_aligned_dim << " index_dim: " << warmup_dim
                   << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
            throw diskann::ANNException(stream.str(), -1);
        }
    }
    else
    {
        warmup = generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
    }
    return warmup;
}

/***************************************************
    Support for Merging Many Vamana Indices
 ***************************************************/

void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs)
{
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *)&npts32, sizeof(uint32_t));
    reader.read((char *)&dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t)npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t))
    {
        std::stringstream stream;
        stream << "Error reading idmap file. Check if the file is bin file with "
                  "1 dimensional data. Actual: "
               << actual_file_size << ", expected: " << (size_t)npts32 + 2 * sizeof(uint32_t) << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *)ivecs.data(), ((size_t)npts32) * sizeof(uint32_t));
    reader.close();
}

int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                 const std::string &idmaps_suffix, const uint64_t nshards, uint32_t max_degree,
                 const std::string &output_vamana, const std::string &medoids_file, bool use_filters,
                 const std::string &labels_to_medoids_file)
{
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
        read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // will merge all the labels to medoids files of each shard into one
    // combined file
    if (use_filters)
    {
        std::unordered_map<uint32_t, std::vector<uint32_t>> global_label_to_medoids;

        for (size_t i = 0; i < nshards; i++)
        {
            std::ifstream mapping_reader;
            std::string map_file = vamana_names[i] + "_labels_to_medoids.txt";
            mapping_reader.open(map_file);

            std::string line, token;
            uint32_t line_cnt = 0;

            while (std::getline(mapping_reader, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t medoid = 0;
                uint32_t label = 0;
                while (std::getline(iss, token, ','))
                {
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

                    uint32_t token_as_num = std::stoul(token);

                    if (cnt == 0)
                        label = token_as_num;
                    else
                        medoid = token_as_num;
                    cnt++;
                }
                global_label_to_medoids[label].push_back(idmaps[i][medoid]);
                line_cnt++;
            }
            mapping_reader.close();
        }

        std::ofstream mapping_writer(labels_to_medoids_file);
        assert(mapping_writer.is_open());
        for (auto iter : global_label_to_medoids)
        {
            mapping_writer << iter.first << ", ";
            auto &vec = iter.second;
            for (uint32_t idx = 0; idx < vec.size() - 1; idx++)
            {
                mapping_writer << vec[idx] << ", ";
            }
            mapping_writer << vec[vec.size() - 1] << std::endl;
        }
        mapping_writer.close();
    }

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (size_t i = 0; i < nshards; i++)
    {
        vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_FOR_CACHED_IO);
        size_t expected_file_size;
        vamana_readers[i].read((char *)&expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana, BUFFER_SIZE_FOR_CACHED_IO);

    size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
                                                     // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write((char *)&merged_index_size,
                               sizeof(uint64_t)); // we will overwrite the index size at the end

    uint32_t output_width = max_degree;
    uint32_t max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
    for (auto &reader : vamana_readers)
    {
        uint32_t input_width;
        reader.read((char *)&input_width, sizeof(uint32_t));
        max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    diskann::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

    merged_vamana_writer.write((char *)&output_width, sizeof(uint32_t));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;
    medoid_writer.write((char *)&nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *)&one_val, sizeof(uint32_t));

    uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
                                      // indices is supported only for bulk indices without frozen point.
                                      // Hence the final index will also not have any frozen points.
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        uint32_t medoid;
        // read medoid
        vamana_readers[shard].read((char *)&medoid, sizeof(uint32_t));
        vamana_readers[shard].read((char *)&vamana_index_frozen, sizeof(uint64_t));
        assert(vamana_index_frozen == false);
        // rename medoid
        medoid = idmaps[shard][medoid];

        medoid_writer.write((char *)&medoid, sizeof(uint32_t));
        // write renamed medoid
        if (shard == (nshards - 1)) //--> uncomment if running hierarchical
            merged_vamana_writer.write((char *)&medoid, sizeof(uint32_t));
    }
    merged_vamana_writer.write((char *)&merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
            // write into merged ofstream
            merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
            merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
            merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            nnbrs = 0;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        vamana_readers[shard_id].read((char *)&shard_nnbrs, sizeof(uint32_t));

        if (shard_nnbrs == 0)
        {
            diskann::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs" << std::endl;
        }

        std::vector<uint32_t> shard_nhood(shard_nnbrs);
        if (shard_nnbrs > 0)
            vamana_readers[shard_id].read((char *)shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < shard_nnbrs; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
    // write into merged ofstream
    merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
    if (nnbrs > 0)
    {
        merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
    }
    merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    diskann::cout << "Expected size: " << merged_index_size << std::endl;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *)&merged_index_size, sizeof(uint64_t));

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}

// TODO: Make this a streaming implementation to avoid exceeding the memory
// budget
/* If the number of filters per point N exceeds the graph degree R,
  then it is difficult to have edges to all labels from this point.
  This function break up such dense points to have only a threshold of maximum
  T labels per point  It divides one graph nodes to multiple nodes and append
  the new nodes at the end. The dummy map contains the real graph id of the
  new nodes added to the graph */
template <typename T>
void breakup_dense_points(const std::string data_file, const std::string labels_file, uint32_t density,
                          const std::string out_data_file, const std::string out_labels_file,
                          const std::string out_metadata_file)
{
    std::string token, line;
    std::ifstream labels_stream(labels_file);
    T *data;
    uint64_t npts, ndims;
    diskann::load_bin<T>(data_file, data, npts, ndims);

    std::unordered_map<uint32_t, uint32_t> dummy_pt_ids;
    uint32_t next_dummy_id = (uint32_t)npts;

    uint32_t point_cnt = 0;

    std::vector<std::vector<uint32_t>> labels_per_point;
    labels_per_point.resize(npts);

    uint32_t dense_pts = 0;
    if (labels_stream.is_open())
    {
        while (getline(labels_stream, line))
        {
            std::stringstream iss(line);
            uint32_t lbl_cnt = 0;
            uint32_t label_host = point_cnt;
            while (getline(iss, token, ','))
            {
                if (lbl_cnt == density)
                {
                    if (label_host == point_cnt)
                        dense_pts++;
                    label_host = next_dummy_id;
                    labels_per_point.resize(next_dummy_id + 1);
                    dummy_pt_ids[next_dummy_id] = (uint32_t)point_cnt;
                    next_dummy_id++;
                    lbl_cnt = 0;
                }
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                uint32_t token_as_num = std::stoul(token);
                labels_per_point[label_host].push_back(token_as_num);
                lbl_cnt++;
            }
            point_cnt++;
        }
    }
    diskann::cout << "fraction of dense points with >= " << density << " labels = " << (float)dense_pts / (float)npts
                  << std::endl;

    if (labels_per_point.size() != 0)
    {
        diskann::cout << labels_per_point.size() << " is the new number of points" << std::endl;
        std::ofstream label_writer(out_labels_file);
        assert(label_writer.is_open());
        for (uint32_t i = 0; i < labels_per_point.size(); i++)
        {
            for (uint32_t j = 0; j < (labels_per_point[i].size() - 1); j++)
            {
                label_writer << labels_per_point[i][j] << ",";
            }
            if (labels_per_point[i].size() != 0)
                label_writer << labels_per_point[i][labels_per_point[i].size() - 1];
            label_writer << std::endl;
        }
        label_writer.close();
    }

    if (dummy_pt_ids.size() != 0)
    {
        diskann::cout << dummy_pt_ids.size() << " is the number of dummy points created" << std::endl;

        T *ptr = (T *)std::realloc((void *)data, labels_per_point.size() * ndims * sizeof(T));
        if (ptr == nullptr)
        {
            diskann::cerr << "Realloc failed while creating dummy points" << std::endl;
            free(data);
            data = nullptr;
            throw new diskann::ANNException("Realloc failed while expanding data.", -1, __FUNCTION__, __FILE__,
                                            __LINE__);
        }
        else
        {
            data = ptr;
        }

        std::ofstream dummy_writer(out_metadata_file);
        assert(dummy_writer.is_open());
        for (auto i = dummy_pt_ids.begin(); i != dummy_pt_ids.end(); i++)
        {
            dummy_writer << i->first << "," << i->second << std::endl;
            std::memcpy(data + i->first * ndims, data + i->second * ndims, ndims * sizeof(T));
        }
        dummy_writer.close();
    }

    diskann::save_bin<T>(out_data_file, data, labels_per_point.size(), ndims);
}

void extract_shard_labels(const std::string &in_label_file, const std::string &shard_ids_bin,
                          const std::string &shard_label_file)
{ // assumes ith row is for ith
  // point in labels file
    diskann::cout << "Extracting labels for shard" << std::endl;

    uint32_t *ids = nullptr;
    uint64_t num_ids, tmp_dim;
    diskann::load_bin(shard_ids_bin, ids, num_ids, tmp_dim);

    uint32_t counter = 0, shard_counter = 0;
    std::string cur_line;

    std::ifstream label_reader(in_label_file);
    std::ofstream label_writer(shard_label_file);
    assert(label_reader.is_open());
    assert(label_reader.is_open());
    if (label_reader && label_writer)
    {
        while (std::getline(label_reader, cur_line))
        {
            if (shard_counter >= num_ids)
            {
                break;
            }
            if (counter == ids[shard_counter])
            {
                label_writer << cur_line << "\n";
                shard_counter++;
            }
            counter++;
        }
    }
    if (ids != nullptr)
        delete[] ids;
}

template <typename T, typename LabelT>
int build_merged_vamana_index(std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R,
                              double sampling_rate, double ram_budget, std::string mem_index_path,
                              std::string medoids_file, std::string centroids_file, size_t build_pq_bytes, bool use_opq,
                              uint32_t num_threads, bool use_filters, const std::string &label_file,
                              const std::string &labels_to_medoids_file, const std::string &universal_label,
                              const uint32_t Lf)
{
    size_t base_num, base_dim;
    diskann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, (uint32_t)base_dim, sizeof(T), R);

    // TODO: Make this honest when there is filter support
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024)
    {
        diskann::cout << "Full index fits in RAM budget, should consume at most "
                      << full_index_ram / (1024 * 1024 * 1024) << "GiBs, so building in one shot" << std::endl;

        diskann::IndexWriteParameters paras = diskann::IndexWriteParametersBuilder(L, R)
                                                  .with_filter_list_size(Lf)
                                                  .with_saturate_graph(!use_filters)
                                                  .with_num_threads(num_threads)
                                                  .build();
        using TagT = uint32_t;
        diskann::Index<T, TagT, LabelT> _index(compareMetric, base_dim, base_num,
                                               std::make_shared<diskann::IndexWriteParameters>(paras), nullptr,
                                               defaults::NUM_FROZEN_POINTS_STATIC, false, false, false,
                                               build_pq_bytes > 0, build_pq_bytes, use_opq, use_filters);
        if (!use_filters)
            _index.build(base_file.c_str(), base_num);
        else
        {
            if (universal_label != "")
            { //  indicates no universal label
                LabelT unv_label_as_num = 0;
                _index.set_universal_label(unv_label_as_num);
            }
            _index.build_filtered_index(base_file.c_str(), label_file, base_num);
        }
        _index.save(mem_index_path.c_str());

        if (use_filters)
        {
            // need to copy the labels_to_medoids file to the specified input
            // file
            std::remove(labels_to_medoids_file.c_str());
            std::string mem_labels_to_medoid_file = mem_index_path + "_labels_to_medoids.txt";
            copy_file(mem_labels_to_medoid_file, labels_to_medoids_file);
            std::remove(mem_labels_to_medoid_file.c_str());
        }

        std::remove(medoids_file.c_str());
        std::remove(centroids_file.c_str());
        return 0;
    }

    // where the universal label is to be saved in the final graph
    std::string final_index_universal_label_file = mem_index_path + "_universal_label.txt";

    std::string merged_index_prefix = mem_index_path + "_tempFiles";

    Timer timer;
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);
    diskann::cout << timer.elapsed_seconds_for_step("partitioning data ") << std::endl;

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    timer.reset();
    for (int p = 0; p < num_parts; p++)
    {
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
        MallocExtension::instance()->ReleaseFreeMemory();
#endif

        std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";

        std::string shard_ids_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";

        std::string shard_labels_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_labels.txt";

        retrieve_shard_data_from_ids<T>(base_file, shard_ids_file, shard_base_file);

        std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

        diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L, 2 * R / 3)
                                                              .with_filter_list_size(Lf)
                                                              .with_saturate_graph(false)
                                                              .with_num_threads(num_threads)
                                                              .build();

        uint64_t shard_base_dim, shard_base_pts;
        get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);

        diskann::Index<T> _index(compareMetric, shard_base_dim, shard_base_pts,
                                 std::make_shared<diskann::IndexWriteParameters>(low_degree_params), nullptr,
                                 defaults::NUM_FROZEN_POINTS_STATIC, false, false, false, build_pq_bytes > 0,
                                 build_pq_bytes, use_opq);
        if (!use_filters)
        {
            _index.build(shard_base_file.c_str(), shard_base_pts);
        }
        else
        {
            diskann::extract_shard_labels(label_file, shard_ids_file, shard_labels_file);
            if (universal_label != "")
            { //  indicates no universal label
                LabelT unv_label_as_num = 0;
                _index.set_universal_label(unv_label_as_num);
            }
            _index.build_filtered_index(shard_base_file.c_str(), shard_labels_file, shard_base_pts);
        }
        _index.save(shard_index_file.c_str());
        // copy universal label file from first shard to the final destination
        // index, since all shards anyway share the universal label
        if (p == 0)
        {
            std::string shard_universal_label_file = shard_index_file + "_universal_label.txt";
            if (universal_label != "")
            {
                copy_file(shard_universal_label_file, final_index_universal_label_file);
            }
        }

        std::remove(shard_base_file.c_str());
    }
    diskann::cout << timer.elapsed_seconds_for_step("building indices on shards") << std::endl;

    timer.reset();
    diskann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file, use_filters,
                          labels_to_medoids_file);
    diskann::cout << timer.elapsed_seconds_for_step("merging indices") << std::endl;

    // delete tempFiles
    for (int p = 0; p < num_parts; p++)
    {
        std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
        std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
        std::string shard_labels_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_labels.txt";
        std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
        std::string shard_index_file_data = shard_index_file + ".data";

        std::remove(shard_base_file.c_str());
        std::remove(shard_id_file.c_str());
        std::remove(shard_index_file.c_str());
        std::remove(shard_index_file_data.c_str());
        if (use_filters)
        {
            std::string shard_index_label_file = shard_index_file + "_labels.txt";
            std::string shard_index_univ_label_file = shard_index_file + "_universal_label.txt";
            std::string shard_index_label_map_file = shard_index_file + "_labels_to_medoids.txt";
            std::remove(shard_labels_file.c_str());
            std::remove(shard_index_label_file.c_str());
            std::remove(shard_index_label_map_file.c_str());
            std::remove(shard_index_univ_label_file.c_str());
        }
    }
    return 0;
}

// General purpose support for DiskANN interface

// optimizes the beamwidth to maximize QPS for a given L_search subject to
// 99.9 latency not blowing up
template <typename T, typename LabelT>
uint32_t optimize_beamwidth(std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> &pFlashIndex, T *tuning_sample,
                            uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L,
                            uint32_t nthreads, uint32_t start_bw)
{
    uint32_t cur_bw = start_bw;
    double max_qps = 0;
    uint32_t best_bw = start_bw;
    bool stop_flag = false;

    while (!stop_flag)
    {
        std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
        std::vector<float> tuning_sample_result_dists(tuning_sample_num, 0);
        diskann::QueryStats *stats = new diskann::QueryStats[tuning_sample_num];

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
        for (int64_t i = 0; i < (int64_t)tuning_sample_num; i++)
        {
            pFlashIndex->page_search(tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
                                            tuning_sample_result_ids_64.data() + (i * 1),
                                            tuning_sample_result_dists.data() + (i * 1), cur_bw, false, stats + i);
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0f * (float)tuning_sample_num) / (1.0f * (float)diff.count());

        double lat_999 = diskann::get_percentile_stats<float>(
            stats, tuning_sample_num, 0.999f, [](const diskann::QueryStats &stats) { return stats.total_us; });

        double mean_latency = diskann::get_mean_stats<float>(
            stats, tuning_sample_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        if (qps > max_qps && lat_999 < (15000) + mean_latency * 2)
        {
            max_qps = qps;
            best_bw = cur_bw;
            cur_bw = (uint32_t)(std::ceil)((float)cur_bw * 1.1f);
        }
        else
        {
            stop_flag = true;
        }
        if (cur_bw > 64)
            stop_flag = true;

        delete[] stats;
    }
    return best_bw;
}

/***************************************************
    Logic of pageann graph construction starts here
 ***************************************************/

/**
 * @brief Merge vectors into pages using graph-topology-based clustering.
 *
 * This function implements a greedy clustering algorithm that groups vectors likely to be
 * accessed together during graph traversal. Starting from an unmerged vector, it performs
 * multi-hop breadth-first expansion along the graph edges to find nearby neighbors and
 * groups them into a single page. This reduces random I/O during search by co-locating
 * vectors that are topologically close in the graph structure.
 *
 * Algorithm:
 * 1. Pick an unmerged starting vector
 * 2. Perform R-hop BFS to collect candidate neighbors
 * 3. Select up to nnodes_per_sector unmerged candidates for the page
 * 4. Repeat until all vectors are assigned to pages
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param nnodes_per_sector Maximum number of vectors per page
 * @param graph_store Loaded Vamana graph topology
 * @param data_store Loaded raw vector dataset
 * @param npts Total number of vectors
 * @param ndim Vector dimensionality
 * @param index_prefix_path Output path prefix for auxiliary files
 * @param R Maximum degree of Vamana graph (used for BFS depth estimation)
 * @param num_pq_chunk Number of PQ chunks
 * @param mergedNodes [Output] mergedNodes[pageID] = list of vector IDs in that page
 * @param nodeToMegaNodeMap [Output] nodeToMegaNodeMap[vectorID] = pageID
 * @param new_to_original_map [Output] Mapping from new ordering to original IDs
 * @return Total number of pages created
 */
template <typename T>
size_t mergeNodesIntoMegaNodes(const uint64_t megaNode_capacity, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<T>> data_store, const uint32_t npts, const uint32_t ndim, const std::string index_prefix_path, const uint32_t R, const uint32_t num_pq_chunk, std::vector<std::vector<uint32_t>>& mergedNodes, std::vector<uint32_t>& nodeToMegaNodeMap, std::vector<uint32_t>& new_to_original_map)
{
    const uint64_t expected_num_megaNodes = (npts + megaNode_capacity - 1) / megaNode_capacity;
    diskann::cout << "Section Length: " << defaults::SECTOR_LEN << "B" << std::endl;
    diskann::cout << "Expected number of mega nodes in total: " << expected_num_megaNodes << std::endl;
    
    uint32_t megaNodeCount = 0;
    uint32_t currNode = 0;
    uint32_t nextNode = -1;

    mergedNodes.reserve(expected_num_megaNodes);
    nodeToMegaNodeMap.reserve(npts);
    nodeToMegaNodeMap.assign(npts, std::numeric_limits<uint32_t>::max());  // Fill with -1

    std::vector<uint8_t> unmergedBitmap; // size = N
    unmergedBitmap.assign(npts, 1);  // every node is initially unmerged
    uint32_t last_unmerged_min = 0;      // smallest possible unmerged ID
    uint32_t unmergedCount = npts;      // all are unmerged at the start

    auto mark_merged = [&](uint32_t id) {
        if (unmergedBitmap[id]) {
            unmergedBitmap[id] = 0;
            unmergedCount--;
            if (id == last_unmerged_min) {
                while (last_unmerged_min < unmergedBitmap.size() &&
                    unmergedBitmap[last_unmerged_min] == 0) {
                    last_unmerged_min++;
                }
            }
        }
    };

    auto is_unmerged = [&](uint32_t id) {
        return unmergedBitmap[id] != 0;
    };

    auto get_next_unmerged = [&]() {
        return last_unmerged_min;
    };

    auto has_unmerged_left = [&]() {
        return unmergedCount > 0;
    };

    size_t num_hops_initial = 2;
    uint32_t numInitialNodes = R * R;
    uint32_t maxNumCandidatesForMaxHop = 1 + R + numInitialNodes;
    while (numInitialNodes < 500 && npts < 1000000000){
        numInitialNodes *= R;
        num_hops_initial++;
        maxNumCandidatesForMaxHop += numInitialNodes;
    }
    diskann::cout << "Num of "<< num_hops_initial << " hops of nodes: " << maxNumCandidatesForMaxHop << std::endl;

    size_t log_interval = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();

//pre-allocated reuse data
    std::vector<uint32_t> groupedNodes;
    groupedNodes.reserve(megaNode_capacity);
    tsl::robin_set<uint32_t> candidate_nbr_set;
    candidate_nbr_set.reserve(maxNumCandidatesForMaxHop); 
    std::vector<uint32_t> new_nbrs_candidates;
    new_nbrs_candidates.reserve(numInitialNodes); 

    while (has_unmerged_left()) {
        groupedNodes.clear();
        candidate_nbr_set.clear();
        new_nbrs_candidates.clear();

        //if this is the last page, we just put the rest unMergedNodesSet within this page
        if (unmergedCount <= megaNode_capacity){
            while(has_unmerged_left()){
                    uint32_t id = get_next_unmerged();
                    mark_merged(id);
                    groupedNodes.push_back(id);         
                    nodeToMegaNodeMap[id] = megaNodeCount;   
            }

            mergedNodes.push_back(groupedNodes);
            megaNodeCount++;
            break;
        }
           
        mark_merged(currNode);
        //diskann::cout << "part 0" << std::endl;
        int hops = num_hops_initial;
        std::vector<uint32_t> frontier = {currNode};//nbrs to expand
        
        for (int hop = 1; hop <= hops; ++hop) {
            new_nbrs_candidates.clear();  // reuse buffer
            //diskann::cout << "Hop: " << hop << std::endl;
            for (const auto& v : frontier){
                auto nbrs = graph_store->get_ooc_neighbours(v);
                for (const auto& nbr : nbrs) {
                    if (candidate_nbr_set.insert(nbr).second) {//inserting successfully means it has not been added as frontier before
                        new_nbrs_candidates.push_back(nbr);
                    }
                }
            }
            // Update new nbrs to expand for next hop
            frontier.swap(new_nbrs_candidates);   // avoids realloc/copy
        }
        //diskann::cout << "part 1" << std::endl;

        //remove all alreadyMergedNodes first --- because this version has no tolerance of duplicates
        std::vector<uint32_t> candidate_nbrs;
        //this could be biggest overhead if not handle well!!!!!
        for (const auto& n : candidate_nbr_set) {
            if (is_unmerged(n)) {
                candidate_nbrs.push_back(n);
            } 
        }
        //diskann::cout << "part 2" << std::endl;
        auto distances = data_store->computeDist(currNode, candidate_nbrs);//max size: 64-1 or 64*64-1
        //diskann::cout << "part 3" << std::endl;
        std::vector<std::pair<uint32_t, float>> id_dist_pairs;
        for (uint32_t i = 0; i < candidate_nbrs.size(); ++i) {
            id_dist_pairs.emplace_back(candidate_nbrs[i], distances[i]);
        }

        std::sort(id_dist_pairs.begin(), id_dist_pairs.end(), 
                [](const auto& a, const auto& b) { return a.second < b.second; });

        groupedNodes.push_back(currNode);
        nodeToMegaNodeMap[currNode] = megaNodeCount;
        
        for (const auto& id_dist : id_dist_pairs) {
            uint32_t id = id_dist.first;

            if (groupedNodes.size() == megaNode_capacity){
                ///MARK: actually we MAY update the new ID here;
                ///MARK: may not get updated cuz there might not extra node withih these n hops
                nextNode = id;
                break;
            }

            groupedNodes.push_back(id);
            nodeToMegaNodeMap[id] = megaNodeCount;
            mark_merged(id);
        }
        
        while (groupedNodes.size() < megaNode_capacity){    
            uint32_t id = get_next_unmerged();
            mark_merged(id);
            groupedNodes.push_back(id);
            nodeToMegaNodeMap[id] = megaNodeCount;
        }

        mergedNodes.push_back(groupedNodes);
        megaNodeCount++;

        if (megaNodeCount % log_interval == 0){
            float percent = 100.0f * megaNodeCount / expected_num_megaNodes;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            diskann::cout << "\rProgress: "
            << std::setw(6) << std::fixed << std::setprecision(2) << percent << "% ("<< megaNodeCount << "/" << expected_num_megaNodes << "), "
            << "time used: " << static_cast<int>(elapsed) << "s"
            << std::flush;
        }

        //there are two cases where nextNode is still -1: 1) no extra node left within maxHops; 2) no enough even candidates left within the maxHops
        if (nextNode == -1){
            currNode = get_next_unmerged();
        }else{
            currNode = nextNode;
        }
        
        nextNode = -1;//flag the nextNode to -1
        ///MARK: continue the iteration by re-assigning currNode instead of always relying on popping from unMergedNodesSet
        //-1 means: there is no node within the searching scope, and we get next node from the left node
    }//end of while loop
    std::cout <<std::endl;
    std::cout << "Actually mega nodes Count: " << mergedNodes.size() << std::endl;
    std::cout << "NodeToMegaNodeMap size: " << nodeToMegaNodeMap.size() << std::endl;

    // get two-way maps
    std::vector<uint32_t> original_to_new(npts, UINT32_MAX); // Initialize with invalid IDs
    
    new_to_original_map.reserve(npts); 
    
    for (const auto &megaNode : mergedNodes) {
        for (uint32_t id : megaNode) {
            if (new_to_original_map.size() >= npts) break;
            original_to_new[id] = static_cast<uint32_t>(new_to_original_map.size());
            new_to_original_map.push_back(id);
        }
        if (new_to_original_map.size() >= npts) break;
    }

    int npts_i32 = static_cast<int>(npts);
    int dim_i32 = 1;  // Since each element is a single uint32_t value
    const std::string old_to_new_map_file = index_prefix_path + "_original_to_new_ids_map.bin";
    std::ofstream old_to_new_map_write(old_to_new_map_file, std::ios::binary);
    if (!old_to_new_map_write) {
        std::cerr << "Error opening file for writing: " << old_to_new_map_file << std::endl;
        return -1;
    }
    old_to_new_map_write.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));  // Write npts
    old_to_new_map_write.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));   // Write dim (1 for each uint32_t)
    old_to_new_map_write.write(reinterpret_cast<char*>(original_to_new.data()), original_to_new.size() * sizeof(uint32_t));
    old_to_new_map_write.close();
    std::cout << "Finish writing old_id to new_id map file " << old_to_new_map_file << std::endl;

//writing the map maping new id to old idea
    const std::string new_to_old_map_file = index_prefix_path + "_new_to_old_ids_map.bin";
    std::ofstream new_to_old_map_write(new_to_old_map_file, std::ios::binary);
    if (!new_to_old_map_write) {
        std::cerr << "Error opening file for writing: " << new_to_old_map_file << std::endl;
        return -1;
    }
    new_to_old_map_write.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));  // Write npts
    new_to_old_map_write.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));   // Write dim (1 for each uint32_t)
    new_to_old_map_write.write(reinterpret_cast<char*>(new_to_original_map.data()), new_to_original_map.size() * sizeof(uint32_t));
    new_to_old_map_write.close();
    std::cout << "Finish writing new_id to old_id map file " << new_to_old_map_file << std::endl;

    diskann::cout << "Saving the final " << megaNodeCount << " merged mega nodes" <<  std::endl;

    return megaNodeCount;
}

/**
 * @brief Merge nodes into mega nodes (pages) using greedy search for candidate collection.
 *
 * This version uses greedy beam search instead of n-hop BFS to collect candidates.
 * Starting from the current vector as entry point, it searches with list size L
 * to find the closest candidates. The top R unmerged candidates are added to the page,
 * and the (R+1)-th closest unmerged candidate becomes the next vector to build around.
 *
 * @tparam T Data type of vectors
 * @param megaNode_capacity Number of vectors per page
 * @param graph_store Graph store containing the Vamana graph
 * @param data_store Data store for distance computation
 * @param npts Total number of points
 * @param ndim Number of dimensions
 * @param index_prefix_path Prefix path for output files
 * @param R Number of closest candidates to add per iteration (page degree)
 * @param L Search list size for greedy search (must be >= R + 1)
 * @param num_pq_chunk Number of PQ chunks (unused, kept for API compatibility)
 * @param mergedNodes Output: pages containing vector IDs
 * @param nodeToMegaNodeMap Output: mapping from vector ID to page ID
 * @param new_to_original_map Output: mapping from new reordered ID to original ID
 * @return Number of pages created
 */
template <typename T>
size_t mergeNodesIntoMegaNodesGreedy(
    const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store,
    std::shared_ptr<InMemOOCDataStore<T>> data_store,
    const uint32_t npts,
    const std::string index_prefix_path,
    const uint32_t L,
    std::vector<std::vector<uint32_t>>& mergedNodes,
    std::vector<uint32_t>& nodeToMegaNodeMap,
    std::vector<uint32_t>& new_to_original_map)
{
    // Validate L >= megaNode_capacity (need enough candidates to fill a page + 1 for nextNode)
    if (L < megaNode_capacity)
    {
        std::cerr << "Error: L must be >= megaNode_capacity. L=" << L << ", megaNode_capacity=" << megaNode_capacity << std::endl;
        return 0;
    }

    const uint64_t expected_num_megaNodes = (npts + megaNode_capacity - 1) / megaNode_capacity;
    diskann::cout << "Section Length: " << defaults::SECTOR_LEN << "B" << std::endl;
    diskann::cout << "Expected number of mega nodes in total: " << expected_num_megaNodes << std::endl;
    diskann::cout << "Using greedy search with L=" << L << std::endl;

    uint32_t megaNodeCount = 0;
    uint32_t currNode = 0;
    uint32_t nextNode = UINT32_MAX;

    mergedNodes.reserve(expected_num_megaNodes);
    nodeToMegaNodeMap.reserve(npts);
    nodeToMegaNodeMap.assign(npts, std::numeric_limits<uint32_t>::max());

    // Bitmap for tracking merged status
    std::vector<uint8_t> unmergedBitmap;
    unmergedBitmap.assign(npts, 1);  // every node is initially unmerged
    uint32_t last_unmerged_min = 0;
    uint32_t unmergedCount = npts;

    auto mark_merged = [&](uint32_t id) {
        if (unmergedBitmap[id]) {
            unmergedBitmap[id] = 0;
            unmergedCount--;
            if (id == last_unmerged_min) {
                while (last_unmerged_min < unmergedBitmap.size() && unmergedBitmap[last_unmerged_min] == 0) {
                    last_unmerged_min++;
                }
            }
        }
    };

    auto is_unmerged = [&](uint32_t id) {
        return unmergedBitmap[id] != 0;
    };

    auto get_next_unmerged = [&]() {
        return last_unmerged_min;
    };

    auto has_unmerged_left = [&]() {
        return unmergedCount > 0;
    };

    size_t log_interval = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Pre-allocated reusable data structures
    std::vector<uint32_t> groupedNodes;
    groupedNodes.reserve(megaNode_capacity);

    while (has_unmerged_left())
    {
        groupedNodes.clear();

        // If this is the last page, just put all remaining unmerged nodes in it
        if (unmergedCount <= megaNode_capacity)
        {
            while (has_unmerged_left())
            {
                uint32_t id = get_next_unmerged();
                mark_merged(id);
                groupedNodes.push_back(id);
                nodeToMegaNodeMap[id] = megaNodeCount;
            }

            mergedNodes.push_back(groupedNodes);
            megaNodeCount++;
            break;
        }

        // Mark current node as merged and add to page
        mark_merged(currNode);
        groupedNodes.push_back(currNode);
        nodeToMegaNodeMap[currNode] = megaNodeCount;

        // Greedy beam search starting from currNode with early termination
        tsl::robin_set<uint32_t> visited;
        visited.reserve(L * 2);
        NeighborPriorityQueue candidate_queue(L);

        // Start from currNode itself (distance 0)
        candidate_queue.insert(Neighbor(currNode, 0.0f));
        visited.insert(currNode);

        // Number of top candidates that need to stabilize before early termination
        // We need megaNode_capacity for current page + 1 for nextNode
        const size_t convergence_threshold = std::min((size_t)(megaNode_capacity * 5), (size_t)L);

        // Greedy beam search with early termination
        while (candidate_queue.has_unexpanded_node())
        {
            // Early termination: cur() points to first unexpanded node
            // If cur() >= convergence_threshold, all top candidates are expanded and stable
            if (candidate_queue.size() >= convergence_threshold && candidate_queue.cur() >= convergence_threshold)
            {
                break;  // Early termination - top candidates are stable
            }

            Neighbor current = candidate_queue.closest_unexpanded();
            auto nbrs = graph_store->get_ooc_neighbours(current.id);
            for (uint32_t nbr_id : nbrs)
            {
                if (visited.insert(nbr_id).second)
                {
                    float dist = data_store->get_distance(currNode, nbr_id);
                    candidate_queue.insert(Neighbor(nbr_id, dist));
                }
            }
        }

        // Collect unmerged candidates sorted by distance
        // candidate_queue is already sorted by distance
        std::vector<uint32_t> unmerged_candidates;
        unmerged_candidates.reserve(L);

        for (size_t i = 0; i < candidate_queue.size(); i++)
        {
            uint32_t cand_id = candidate_queue[i].id;
            if (cand_id != currNode && is_unmerged(cand_id))
            {
                unmerged_candidates.push_back(cand_id);
            }
        }

        // Fill page with unmerged candidates from search (up to megaNode_capacity - 1 since currNode is already added)
        size_t candidates_used = 0;
        for (size_t i = 0; i < unmerged_candidates.size() && groupedNodes.size() < megaNode_capacity; i++)
        {
            uint32_t id = unmerged_candidates[i];
            groupedNodes.push_back(id);
            nodeToMegaNodeMap[id] = megaNodeCount;
            mark_merged(id);
            candidates_used++;
        }

        // Set nextNode to the first unused candidate (for next page's starting point)
        if (candidates_used < unmerged_candidates.size())
        {
            uint32_t next_candidate = unmerged_candidates[candidates_used];
            if (is_unmerged(next_candidate))
            {
                nextNode = next_candidate;
            }
        }

        // Fill remaining slots with sequential unmerged nodes if search didn't find enough
        while (groupedNodes.size() < megaNode_capacity)
        {
            uint32_t id = get_next_unmerged();
            mark_merged(id);
            groupedNodes.push_back(id);
            nodeToMegaNodeMap[id] = megaNodeCount;
        }

        mergedNodes.push_back(groupedNodes);
        megaNodeCount++;

        if (megaNodeCount % log_interval == 0)
        {
            float percent = 100.0f * megaNodeCount / expected_num_megaNodes;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            diskann::cout << "\rProgress: "
                          << std::setw(6) << std::fixed << std::setprecision(2) << percent << "% ("
                          << megaNodeCount << "/" << expected_num_megaNodes << "), "
                          << "time used: " << static_cast<int>(elapsed) << "s"
                          << std::flush;
        }

        // Determine next node to build around
        if (nextNode == UINT32_MAX || !is_unmerged(nextNode))
        {
            // No valid next node from search, get next unmerged
            currNode = get_next_unmerged();
        }
        else
        {
            currNode = nextNode;
        }

        nextNode = UINT32_MAX;  // Reset for next iteration
    }

    std::cout << std::endl;
    std::cout << "Actually mega nodes Count: " << mergedNodes.size() << std::endl;
    std::cout << "NodeToMegaNodeMap size: " << nodeToMegaNodeMap.size() << std::endl;

    // Build two-way maps
    std::vector<uint32_t> original_to_new(npts, UINT32_MAX);

    new_to_original_map.reserve(npts);

    for (const auto& megaNode : mergedNodes)
    {
        for (uint32_t id : megaNode)
        {
            if (new_to_original_map.size() >= npts) break;
            original_to_new[id] = static_cast<uint32_t>(new_to_original_map.size());
            new_to_original_map.push_back(id);
        }
        if (new_to_original_map.size() >= npts) break;
    }

    // Write original_to_new map
    int npts_i32 = static_cast<int>(npts);
    int dim_i32 = 1;
    const std::string old_to_new_map_file = index_prefix_path + "_original_to_new_ids_map.bin";
    std::ofstream old_to_new_map_write(old_to_new_map_file, std::ios::binary);
    if (!old_to_new_map_write)
    {
        std::cerr << "Error opening file for writing: " << old_to_new_map_file << std::endl;
        return 0;
    }
    old_to_new_map_write.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));
    old_to_new_map_write.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));
    old_to_new_map_write.write(reinterpret_cast<char*>(original_to_new.data()), original_to_new.size() * sizeof(uint32_t));
    old_to_new_map_write.close();
    std::cout << "Finish writing old_id to new_id map file " << old_to_new_map_file << std::endl;

    // Write new_to_original map
    const std::string new_to_old_map_file = index_prefix_path + "_new_to_old_ids_map.bin";
    std::ofstream new_to_old_map_write(new_to_old_map_file, std::ios::binary);
    if (!new_to_old_map_write)
    {
        std::cerr << "Error opening file for writing: " << new_to_old_map_file << std::endl;
        return 0;
    }
    new_to_old_map_write.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));
    new_to_old_map_write.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));
    new_to_old_map_write.write(reinterpret_cast<char*>(new_to_original_map.data()), new_to_original_map.size() * sizeof(uint32_t));
    new_to_old_map_write.close();
    std::cout << "Finish writing new_id to old_id map file " << new_to_old_map_file << std::endl;

    diskann::cout << "Saving the final " << megaNodeCount << " merged mega nodes" << std::endl;

    return megaNodeCount;
}

/***************************************************
    Logic of laann graph construction starts here
 ***************************************************/

// Helper: greedy beam search on Vamana graph to find candidates for a vector.
// Returns L closest neighbors (original IDs), excluding the query vector itself.
template <typename T>
static void search_for_candidates(
    const uint32_t vector_id,
    std::shared_ptr<InMemOOCGraphStore> graph_store,
    std::shared_ptr<InMemOOCDataStore<T>> data_store,
    const uint32_t L,
    std::vector<Neighbor>& candidates)
{
    candidates.clear();
    tsl::robin_set<uint32_t> visited;
    visited.reserve(L * 2);
    visited.insert(vector_id);

    NeighborPriorityQueue candidate_queue(L);
    for (uint32_t nbr_id : graph_store->get_ooc_neighbours(vector_id)) {
        if (visited.insert(nbr_id).second) {
            candidate_queue.insert(Neighbor(nbr_id, data_store->get_distance(vector_id, nbr_id)));
        }
    }
    while (candidate_queue.has_unexpanded_node()) {
        Neighbor current = candidate_queue.closest_unexpanded();
        for (uint32_t nbr_id : graph_store->get_ooc_neighbours(current.id)) {
            if (visited.insert(nbr_id).second) {
                candidate_queue.insert(Neighbor(nbr_id, data_store->get_distance(vector_id, nbr_id)));
            }
        }
    }
    candidates.reserve(candidate_queue.size());
    for (size_t i = 0; i < candidate_queue.size(); i++)
        candidates.push_back(candidate_queue[i]);
}

/**
 * @brief Create disk layout for PageANN page-based index storage.
 */
template <typename T>
void create_pageann_disk_layout(std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes,
                       const uint64_t nnodes_per_sector, const float pq_cache_ratio,
                       std::shared_ptr<InMemOOCGraphStore> graph_store,
                       std::shared_ptr<InMemOOCDataStore<T>> data_store, const size_t ndims,
                       const uint32_t page_graph_degree, const std::string &pq_compressed_all_nodes_path,
                       const std::string &output_file,
                       const std::vector<std::vector<uint32_t>>& mergedNodes,
                       const std::vector<uint32_t>& nodeToMegaNodeMap,
                       const uint32_t fill_L,
                       bool enable_spare_fill) {
    // ===== STEP 1: Load PQ compressed data into memory =====
    int all_pq_npts, pq_ndims;
    std::ifstream pqAllPointsReader;
    pqAllPointsReader.exceptions(std::ios::badbit | std::ios::failbit);
    pqAllPointsReader.open(pq_compressed_all_nodes_path, std::ios::binary);
    pqAllPointsReader.seekg(0, pqAllPointsReader.beg);
    pqAllPointsReader.read((char *)&all_pq_npts, sizeof(int));
    pqAllPointsReader.read((char *)&pq_ndims, sizeof(int));  // i.e., num_pq_chunks

    diskann::cout << "Number of points of compressed PQ data: " << all_pq_npts << std::endl;
    diskann::cout << "Number of dimensions of compressed PQ data: " << pq_ndims << std::endl;

    size_t total_bytes = static_cast<size_t>(all_pq_npts) * pq_ndims;
    std::cout << "Total bytes to allocate: " << total_bytes << " (" << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;

    // Read all PQ data into memory array
    std::unique_ptr<uint8_t[]> pq_compressed_page_all_points = std::make_unique<uint8_t[]>(total_bytes);
    pqAllPointsReader.read((char*)(pq_compressed_page_all_points.get()), total_bytes * sizeof(uint8_t));
    pqAllPointsReader.close();

    // ===== STEP 2: Calculate metadata and reorder PQ data to match page layout =====
    uint64_t each_node_sapce = (uint64_t)(ndims * sizeof(T));  // Space per vector in bytes
    diskann::cout << "Section Length: " << defaults::SECTOR_LEN << "B" << std::endl;
    diskann::cout << "each_node_sapce: " << each_node_sapce << "B" << std::endl;
    diskann::cout << "nnodes_per_sector: " << nnodes_per_sector << std::endl;

    size_t num_node, npts_64_aligned, ndims_64;
    num_node = (uint64_t)nodeToMegaNodeMap.size();
    npts_64_aligned = (uint64_t)(nnodes_per_sector * mergedNodes.size());  // Total slots in all pages
    ndims_64 = (uint64_t)ndims;
    // Note: num_node (actual vectors) can be different from npts_64_aligned (total page slots)
    diskann::cout << "Number of points in reordered data on Disk: " << npts_64_aligned << std::endl;
    diskann::cout << "Number of dimensions of reordered data: " << ndims << std::endl;
    diskann::cout << "Number of nodes in graph: " << num_node << std::endl;

    // Medoid is the entry point (start node from Vamana index)
    uint64_t medoid = (uint64_t)nodeToMegaNodeMap[graph_store->get_start()];  // PageID containing the medoid
    diskann::cout << "medoid pageID: " << medoid << std::endl;

    // Reorder PQ data to match the new page-based layout
    // Map: old_node_ID -> new_reordered_node_ID -> reordered PQ data
    size_t safe_pq_ndims = static_cast<size_t>(pq_ndims);
    const size_t pq_size = safe_pq_ndims * sizeof(uint8_t);
    reorder_pq_data_buff = std::make_unique<uint8_t[]>(total_bytes);

    for (size_t old_node_ID = 0; old_node_ID < num_node; old_node_ID++){
        uint32_t nbrPageID = nodeToMegaNodeMap[old_node_ID];
        const auto& allNodesWithinThePage = mergedNodes[nbrPageID];
        auto it = std::find(allNodesWithinThePage.begin(), allNodesWithinThePage.end(), old_node_ID);
        if (it == allNodesWithinThePage.end()){
            std::cout << "Node ID " << old_node_ID << " not found in page " << nbrPageID << "\n";
            continue;
        }
        // Calculate new node ID: pageID * vectors_per_page + offset_within_page
        uint32_t new_reordered_node_ID = static_cast<uint32_t>(nbrPageID * nnodes_per_sector + (it - allNodesWithinThePage.begin()));
        memcpy(&reorder_pq_data_buff[new_reordered_node_ID * safe_pq_ndims], &pq_compressed_page_all_points[old_node_ID * safe_pq_ndims], pq_size);
    }

    // ===== STEP 3: Initialize output file and prepare metadata =====
    size_t write_blk_size = 64 * 1024 * 1024;  // Buffered write size (64 MB)
    cached_ofstream diskann_writer(output_file, write_blk_size);

    uint64_t n_sectors_total = mergedNodes.size();
    uint64_t disk_index_file_size = (n_sectors_total + 1) * defaults::SECTOR_LEN;  // +1 for metadata sector

    // Prepare metadata to be written at the beginning of the file
    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back((uint64_t)npts_64_aligned);  // Total number of vector slots
    output_file_meta.push_back((uint64_t)ndims_64);         // Vector dimensionality
    const uint64_t page_size = 4;
    output_file_meta.push_back(page_size);                  // Page size in KB
    output_file_meta.push_back(medoid);                     // Medoid page ID (entry point)
    output_file_meta.push_back(each_node_sapce);            // Bytes per vector
    output_file_meta.push_back(nnodes_per_sector);          // Vectors per page
    output_file_meta.push_back((uint64_t)pq_ndims);         // PQ chunks per vector

    // Reserve first sector for metadata (will be written at the end)
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);  // 4096 bytes
    diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
    size_t log_interval = 1000;

    // Set up pointers to the fixed header fields within the sector buffer
    // Layout: [vectors][num_cached u16][num_uncached u16][cached_IDs][uncached_IDs][uncached_PQ]
    char *sector_num_cached_nbrs_buf = sector_buf.get() + nnodes_per_sector * each_node_sapce;
    char *sector_num_uncached_nbrs_buf = sector_num_cached_nbrs_buf + sizeof(uint16_t);
    char *sector_cached_nbrs_IDs_buf = sector_num_uncached_nbrs_buf + sizeof(uint16_t);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Pre-allocate space for cached PQ nodes set
    if (pq_cache_ratio == 1.0f)
        cached_PQ_nodes.reserve(num_node);
    else
        cached_PQ_nodes.reserve(static_cast<size_t>(num_node * pq_cache_ratio));

    // Fixed per-page split: first target_cached slots → cached, rest → uncached
    // Space guarantee: target_cached*4 + target_uncached*(4+pq_ndims) <= total_space_for_nbrs
    const size_t target_cached = static_cast<size_t>(page_graph_degree * pq_cache_ratio);
    const size_t target_uncached = page_graph_degree - target_cached;

    // Pre-build old_to_new ID map for O(1) lookup (avoids std::find per neighbor)
    std::vector<uint32_t> old_to_new(num_node, UINT32_MAX);
    for (uint32_t page_id = 0; page_id < static_cast<uint32_t>(mergedNodes.size()); page_id++) {
        for (uint32_t k = 0; k < static_cast<uint32_t>(mergedNodes[page_id].size()); k++) {
            uint32_t old_id = mergedNodes[page_id][k];
            if (old_id < num_node) {
                uint32_t new_id = page_id * static_cast<uint32_t>(nnodes_per_sector) + k;
                old_to_new[old_id] = new_id;
                if (pq_cache_ratio == 1.0f)
                    cached_PQ_nodes.insert(new_id);
            }
        }
    }

    // ===== STEP 4: Write page data and topology to disk =====
    uint32_t maxDeg = 0, totalDeg = 0, minDeg = std::numeric_limits<uint32_t>::max();
    size_t num_beam_search_pages = 0;

    {
        std::vector<T> cur_vec(data_store->get_aligned_dim());

        for (uint32_t sector = 0; sector < mergedNodes.size(); sector++) {
            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
            const auto& group = mergedNodes[sector];

            // === Step 4.1: Write vector data to sector buffer ===
            char* sector_nodes_buf = sector_buf.get();
            tsl::robin_set<uint32_t> page_vector_set;
            page_vector_set.reserve(group.size());

            for (auto& id : group) {
                std::memset(cur_vec.data(), 0, data_store->get_aligned_dim() * sizeof(T));
                data_store->get_vector(id, cur_vec.data());
                memcpy(sector_nodes_buf, cur_vec.data(), data_store->get_aligned_dim() * sizeof(T));
                page_vector_set.insert(id);
                sector_nodes_buf += each_node_sapce;
            }

            // === Step 4.2: Merge all direct neighbors from page vectors (LAANN-style) ===
            tsl::robin_set<uint32_t> merged_nbr_set;
            merged_nbr_set.reserve(page_graph_degree * 2);
            std::vector<uint32_t> page_nbrs;
            page_nbrs.reserve(page_graph_degree);
            uint32_t spare = page_graph_degree;

            for (uint32_t k = 0; k < group.size() && spare > 0; k++) {
                for (const auto& nbr_old_id : graph_store->get_ooc_neighbours(group[k])) {
                    if (spare == 0) break;
                    if (page_vector_set.count(nbr_old_id)) continue;
                    if (merged_nbr_set.insert(nbr_old_id).second) {
                        page_nbrs.push_back(old_to_new[nbr_old_id]);
                        spare--;
                    }
                }
            }

            // === Assign cached / uncached — two exclusive paths based on pq_cache_ratio ===

            std::vector<uint32_t> cached_nbrs, uncached_nbrs;

            if (pq_cache_ratio == 1.0f) {
                // Full PQ cache: beam search fills remaining slots, all neighbors cached
                if (enable_spare_fill && spare > 0) {
                    std::vector<Neighbor> candidates;
                    for (size_t vec_idx = 0; vec_idx < group.size() && spare > 0; vec_idx++) {
                        search_for_candidates<T>(group[vec_idx], graph_store, data_store, fill_L, candidates);
                        for (const auto& cand : candidates) {
                            if (spare == 0) break;
                            if (page_vector_set.count(cand.id)) continue;
                            if (merged_nbr_set.insert(cand.id).second) {
                                page_nbrs.push_back(old_to_new[cand.id]);
                                spare--;
                            }
                        }
                    }
                    num_beam_search_pages++;
                }
                cached_nbrs.reserve(page_nbrs.size());
                for (uint32_t new_id : page_nbrs)
                    cached_nbrs.push_back(new_id);
            } else {
                // Partial PQ cache: split direct neighbors — first target_cached → cached, rest → uncached
                cached_nbrs.reserve(std::min(target_cached, page_nbrs.size()));
                uncached_nbrs.reserve(target_uncached);
                for (size_t j = 0; j < page_nbrs.size(); j++) {
                    if (j < target_cached) {
                        cached_nbrs.push_back(page_nbrs[j]);
                        cached_PQ_nodes.insert(page_nbrs[j]);
                    } else {
                        uncached_nbrs.push_back(page_nbrs[j]);
                    }
                }
            }

            // === Step 4.4: Write neighbor metadata and lists to sector buffer ===
            // Sector layout: [vectors][num_cached][num_uncached][cached_IDs][uncached_IDs][uncached_PQ]
            uint16_t num_cached_nbrs = static_cast<uint16_t>(cached_nbrs.size());
            memcpy(sector_num_cached_nbrs_buf, &num_cached_nbrs, sizeof(uint16_t));

            uint16_t num_uncached_nbrs = static_cast<uint16_t>(uncached_nbrs.size());
            memcpy(sector_num_uncached_nbrs_buf, &num_uncached_nbrs, sizeof(uint16_t));

            uint32_t total_num_nbrs = (uint32_t)num_cached_nbrs + (uint32_t)num_uncached_nbrs;
            maxDeg = std::max(maxDeg, total_num_nbrs);
            minDeg = std::min(minDeg, total_num_nbrs);
            totalDeg += total_num_nbrs;

            memcpy(sector_cached_nbrs_IDs_buf, cached_nbrs.data(), cached_nbrs.size() * sizeof(uint32_t));

            char* sector_uncached_nbrs_IDs_buf = sector_cached_nbrs_IDs_buf + cached_nbrs.size() * sizeof(uint32_t);
            memcpy(sector_uncached_nbrs_IDs_buf, uncached_nbrs.data(), uncached_nbrs.size() * sizeof(uint32_t));

            char* sector_uncached_nbrs_pq_buf = sector_uncached_nbrs_IDs_buf + uncached_nbrs.size() * sizeof(uint32_t);
            for (uint32_t new_node_ID : uncached_nbrs) {
                memcpy(sector_uncached_nbrs_pq_buf, &reorder_pq_data_buff[new_node_ID * safe_pq_ndims], pq_size);
                sector_uncached_nbrs_pq_buf += pq_size;
            }

            // === Step 4.5: Flush sector to disk ===
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);

            if (sector % log_interval == 0) {
                uint32_t pageCount = sector + 1;
                float percent = 100.0f * pageCount / mergedNodes.size();
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                diskann::cout << "\rProgress of writing to disk: "
                              << std::setw(6) << std::fixed << std::setprecision(2) << percent << "% ("
                              << pageCount << "/" << mergedNodes.size() << "), "
                              << "time used: " << static_cast<int>(elapsed) << "s"
                              << std::flush;
            }
        }  // End of page loop

        // ===== STEP 5: Print statistics =====
        diskann::cout << "\nMax degree of page graph: " << maxDeg << std::endl;
        diskann::cout << "Min degree of page graph: " << minDeg << std::endl;
        diskann::cout << "Average degree of page graph: " << static_cast<double>(totalDeg) / mergedNodes.size() << std::endl;
        diskann::cout << "Pages using beam search fallback: " << num_beam_search_pages << std::endl;
    }

    diskann::cout << std::endl;
    diskann_writer.close();  // Flush all buffered data to disk

    // Write metadata to the first sector (reserved earlier)
    output_file_meta.push_back((uint64_t)maxDeg);
    output_file_meta.push_back(disk_index_file_size);
    diskann::save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    diskann::cout << "Output disk index file written to " << output_file << std::endl;
}

/**
 * @brief Calculate how many vectors fit in a page given a page degree.
 * Each page has a fixed size (SECTOR_LEN, typically 4KB). This function computes
 * how many vectors can be stored in a page alongside the page-level graph metadata,
 * given the page's outgoing edge count (pageDegree).
 *
 * Page layout:
 * - Vector data: dimension * typeByte * megaNode_capacity
 * - Metadata: 2 * sizeof(uint16_t) for neighbor counts
 * - Cached neighbors: sizeof(uint32_t) * (pageDegree * sampleRatio)
 * - Uncached neighbors: (sizeof(uint32_t) + PQ_size) * (pageDegree * (1 - sampleRatio))
 */
size_t get_pageann_megaNode_capacity(size_t dimension, size_t typeByte, size_t pageDegree, size_t PQ_size, float sampleRatio) {
    size_t minNumCachedNbrs = static_cast<size_t>(pageDegree * sampleRatio);
    size_t numUncachedNbrs = pageDegree - minNumCachedNbrs;
    //sizeOfActualCachedNbrs + sizeOfActualUncachedNbrs      //nbrID
    size_t overhead = sizeof(uint16_t) + sizeof(uint16_t) + (sizeof(uint32_t) + PQ_size) * numUncachedNbrs + sizeof(uint32_t) * minNumCachedNbrs;
    size_t remainingBytes = defaults::SECTOR_LEN - overhead;
    return remainingBytes / (dimension * typeByte);
}


/**
 * @brief Find optimal page degree and vectors-per-page to satisfy min degree constraint.
 *
 * This function solves the optimization problem:
 * - Constraint: avgDegree = pageDegree / megaNode_capacity >= min_degree_per_node
 * - Objective 1: Maximize megaNode_capacity (more vectors per page = fewer pages)
 * - Objective 2: Maximize pageDegree (more edges = better search accuracy)
 *
 * Algorithm:
 * 1. Find minimum pageDegree where avgDegree >= min_degree_per_node
 * 2. Record the corresponding megaNode_capacity as target
 * 3. Increase pageDegree as much as possible while keeping megaNode_capacity = target
 *
 * @param dimension Vector dimensionality
 * @param typeByte Size of data type (4 for float, 1 for int8/uint8)
 * @param PQ_size Number of bytes per PQ-compressed vector
 * @param sampleRatio Fraction of neighbors that are cached
 * @param min_degree_per_node Minimum required average degree per vector
 * @return Pair of (optimal_page_degree, optimal_megaNode_capacity)
 *         Returns (0, 0) if min_degree_per_node is unachievable
 */
std::pair<size_t, size_t> get_optimal_pageann_degree_megaNode_capacity(size_t dimension, size_t typeByte, size_t PQ_size, float sampleRatio, size_t min_degree_per_node) {
    
    size_t pageDegree = min_degree_per_node;
    float avgDegree = 0.0;
    size_t megaNode_capacity = 0;

    while (avgDegree < static_cast<float>(min_degree_per_node)) {
        megaNode_capacity = get_pageann_megaNode_capacity(dimension, typeByte, pageDegree, PQ_size, sampleRatio);
        if (megaNode_capacity <= 0){
            std::cout << "Error: a page cannot host a single vector with the given degree" << std::endl;
            return {0, 0};
        }

        avgDegree = static_cast<float>(pageDegree) / megaNode_capacity;

        pageDegree++;
    }

    ///MARK: megaNode_capacity here is the smallest number meeting the requirement of min_degre_per_node
    ///MARK: we stop here, because we also want the store as many vectors within a page as possible under the guarantee of min degree per node
    size_t target_megaNode_capacity = megaNode_capacity;
    std::cout << "target # vectors per page: " << target_megaNode_capacity << ", page node max degree: " << pageDegree <<"\n";
    std::cout << "Begin to max the degree with this target nnnode_per_page\n";

    // Find the largest pageDegree such numNodes is still target_megaNode_capacity
    megaNode_capacity = get_pageann_megaNode_capacity(dimension, typeByte, pageDegree, PQ_size, sampleRatio); //continue last while loop
    while (megaNode_capacity == target_megaNode_capacity) {
        avgDegree = static_cast<float>(pageDegree) / megaNode_capacity;
        pageDegree++;
        megaNode_capacity = get_pageann_megaNode_capacity(dimension, typeByte, pageDegree, PQ_size, sampleRatio);
    }
    
    pageDegree--;
    std::cout << "Optimal page node max degree: " << pageDegree << ", # vectors per page: " << target_megaNode_capacity << ", max degree per vector: " << avgDegree << "\n";

    return {pageDegree, target_megaNode_capacity};
}

/**
 * @brief Recommend optimal Vamana graph degree (R parameter) for page graph construction.
 *
 * This function helps users determine the appropriate maximum degree (R) for building
 * a Vamana index that will later be converted to a PageANN page-level graph. It computes
 * the optimal page structure parameters and provides recommendations based on the dataset
 * characteristics and memory budget.
 *
 * Output includes:
 * - Recommended Vamana graph degree (R)
 * - Expected page degree and vectors per page
 * - Memory allocation between PQ caching and LSH sampling
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param data_file_to_use Path to raw vector dataset
 * @param min_degree_per_node Minimum average degree per vector in page graph
 * @param num_pq_chunks_32 Number of PQ chunks for compression
 * @param memBudgetInGB Memory budget in GB for search process
 * @param full_ooc If true, assume fully out-of-core mode (no caching)
 */
template <typename T>
void optimal_vamana_graph_degree(const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t num_pq_chunks_32, float memBudgetInGB, bool full_ooc)
{
    size_t points_num, dim;
    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

    float pq_cache_ratio = getCacheRatio(memBudgetInGB, static_cast<size_t>(num_pq_chunks_32), points_num);
    if (pq_cache_ratio < 0){
        std::cout << "Error: memory budget shoud be at least 0.06 GB. Please increase your memory budget." << std::endl;
        return;
    }

    if(full_ooc){
        pq_cache_ratio = 0.0;
    }

    double cached_size = sizeof(uint32_t);
    double uncached_size = sizeof(uint32_t) + static_cast<double>(num_pq_chunks_32);
    double avg_size_per_neighbor = pq_cache_ratio * cached_size + (1.0 - pq_cache_ratio) * uncached_size;
    double available_bytes = static_cast<double>(defaults::SECTOR_LEN) - static_cast<double>(dim * sizeof(T)) - static_cast<double>(sizeof(uint16_t) * 2);//two uint16 meta data for nnbrs
    size_t max_degree = static_cast<size_t>(std::floor(available_bytes / avg_size_per_neighbor));

    if (min_degree_per_node > max_degree){
        std::cout << "Error: a page cannot host a single vector with the given degree. Please decrease your min degree requirement for each vector." << std::endl;
        return;
    }

    auto result2 = get_optimal_pageann_degree_megaNode_capacity(dim, sizeof(T), static_cast<size_t>(num_pq_chunks_32), pq_cache_ratio, static_cast<size_t>(min_degree_per_node));
    uint32_t page_graph_degree = static_cast<uint32_t>(result2.first);
    uint32_t megaNode_capacity = static_cast<uint32_t>(result2.second);

    if (megaNode_capacity <= 0){
        std::cout << "Error: a page cannot host a single vector with the given degree. Please decrease your min degree requirement for each node." << std::endl;
        return;
    }

    // Get integer and fractional parts
    float avgDegree = static_cast<float>(page_graph_degree) / megaNode_capacity;
    uint32_t intPart = static_cast<uint32_t>(avgDegree);
    float fracPart = avgDegree - intPart;
    // Apply your rounding rule
    uint32_t roundedAvgDegree = (fracPart >= 0.9f) ? (intPart + 1) : intPart;
    page_graph_degree = megaNode_capacity * roundedAvgDegree;
    
    std::cout << "Final recommended vamana disk graph max degree: " << roundedAvgDegree;
    //std::cout << ", and its corresponding page-node graph max degree: " << page_graph_degree;
    std::cout << std::endl;
}

/**
 * @brief Generate reorganized PQ data file and calculate memory usage.
 *
 * This function creates a new PQ compressed file where the PQ data is reordered to match
 * the page-based layout. It also computes the total memory footprint for PQ caching during
 * search, which includes:
 * - Cached PQ data (for nodes in cached_PQ_nodes set)
 * - Mapping structure to locate cached PQ data (dense array or sparse map depending on ratio)
 *
 * @param target_pq_cache_ratio Target fraction of PQ data to cache in memory
 * @param points_num Total number of vectors in dataset
 * @param reorder_pq_data_buff Buffer containing reordered PQ data for cached nodes
 * @param cached_PQ_nodes Set of node IDs whose PQ data is cached
 * @param num_pq_chunks_32 Number of PQ chunks per vector
 * @param pq_compressed_reorder_path Output path for reorganized PQ file
 * @return Total memory usage in bytes for PQ caching structures
 */
size_t generate_pageann_pq_data(const float target_pq_cache_ratio, const size_t points_num, std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint32_t num_pq_chunks_32, const std::string &pq_compressed_reorder_path){
    
    std::cout << "Number of cached PQ: " << cached_PQ_nodes.size() << "\n";
    double actual_PQ_cache_ratio = static_cast<double>(cached_PQ_nodes.size()) / points_num;
    std::cout << std::fixed << std::setprecision(4) << "Actual cached PQ ratio: " << actual_PQ_cache_ratio << "\n";
    std::cout << std::fixed << std::setprecision(4) << "Expected cached PQ ratio: " << target_pq_cache_ratio << "\n";

    ///calculate and compare which is more memory-saving? store map or just full pq data
    size_t left_PQ_space = (points_num - cached_PQ_nodes.size()) * num_pq_chunks_32 * sizeof(uint8_t);
    if (target_pq_cache_ratio == 1.0){
        left_PQ_space = 0;
    }
    const size_t INT32_SIZE = sizeof(uint32_t);
    size_t size_cachedNodeID_PQidx_map = 0;
    //decide to use nodeID_PQindex map or array
    bool useID_pqIdx_map = false;
    bool useID_pqIdx_array = false;

    if (actual_PQ_cache_ratio >= 0.25) {
        //use array
        size_cachedNodeID_PQidx_map = INT32_SIZE * points_num;
        useID_pqIdx_array = true;
    } else {
        //use map
        size_cachedNodeID_PQidx_map = 2 * INT32_SIZE * cached_PQ_nodes.size() * 2;
        useID_pqIdx_map = true;
    }

    size_t size_final_cached_PQ = cached_PQ_nodes.size() * num_pq_chunks_32;
    uint32_t num_cached_nodes = static_cast<uint32_t>(cached_PQ_nodes.size());
    uint32_t points_num_32 = static_cast<uint32_t>(points_num);
    if (left_PQ_space <= size_cachedNodeID_PQidx_map) {
        size_final_cached_PQ = points_num * num_pq_chunks_32 * sizeof(uint8_t);
        num_cached_nodes = points_num_32;
        size_cachedNodeID_PQidx_map = 0;
        useID_pqIdx_map = false;
        useID_pqIdx_array = false;
    } 

    ///now we begin to store PQ compress data
    //write the meta data first
    std::ofstream outFile(pq_compressed_reorder_path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file "<< pq_compressed_reorder_path << " for writing.\n";
    }
    //write metadata: number of total points, number of cached points, num_pq_chunks_32, useID_pqIdx_map, useID_pqIdx_array
    outFile.write(reinterpret_cast<const char*>(&points_num_32), sizeof(points_num_32));
    outFile.write(reinterpret_cast<const char*>(&num_cached_nodes), sizeof(num_cached_nodes));
    outFile.write(reinterpret_cast<const char*>(&num_pq_chunks_32), sizeof(num_pq_chunks_32));
    uint8_t useID_pqIdx_map_u8 = useID_pqIdx_map ? 1 : 0;
    uint8_t useID_pqIdx_array_u8 = useID_pqIdx_array ? 1 : 0;
    outFile.write(reinterpret_cast<const char*>(&useID_pqIdx_map_u8), sizeof(useID_pqIdx_map_u8));
    outFile.write(reinterpret_cast<const char*>(&useID_pqIdx_array_u8), sizeof(useID_pqIdx_array_u8));

    //use extra map
    size_t safe_num_pq_chunk = static_cast<size_t>(num_pq_chunks_32);
    if (useID_pqIdx_map){
        //create a map where key is nodeID while the value is the pq_idx
        std::unique_ptr<uint8_t[]> cached_pq_buff_to_write = std::make_unique<uint8_t[]>(cached_PQ_nodes.size() * safe_num_pq_chunk);
        std::vector<uint32_t> pqIdx_nodeID_map;
        pqIdx_nodeID_map.reserve(cached_PQ_nodes.size()); //index of this array is the pq_idx -- when decoding, we need to reverse this key-value relationship
        for(uint32_t new_node_ID : cached_PQ_nodes){
            memcpy(&cached_pq_buff_to_write[pqIdx_nodeID_map.size() * safe_num_pq_chunk], &reorder_pq_data_buff[new_node_ID * safe_num_pq_chunk], safe_num_pq_chunk);
            pqIdx_nodeID_map.push_back(new_node_ID);
        }
        //write cached_pq_buff_to_write into file
        outFile.write(reinterpret_cast<const char*>(cached_pq_buff_to_write.get()), cached_PQ_nodes.size() * safe_num_pq_chunk);
        //write pqIdx_nodeID_map into file
        outFile.write(reinterpret_cast<const char*>(pqIdx_nodeID_map.data()), pqIdx_nodeID_map.size() * sizeof(uint32_t));
    }
    //use extra array
    else if(useID_pqIdx_array){
        //create an array with size of npts, where index is the nodeID while the value is the pq_idx
        std::unique_ptr<uint8_t[]> cached_pq_buff_to_write = std::make_unique<uint8_t[]>(cached_PQ_nodes.size() * safe_num_pq_chunk);
        std::vector<uint32_t> nodeID_pqIdx_map(points_num, std::numeric_limits<uint32_t>::max());
        uint32_t pq_count = 0;
        for(uint32_t new_node_ID : cached_PQ_nodes){
            nodeID_pqIdx_map[new_node_ID] = pq_count;
            memcpy(&cached_pq_buff_to_write[pq_count * safe_num_pq_chunk], &reorder_pq_data_buff[new_node_ID * safe_num_pq_chunk], safe_num_pq_chunk);
            pq_count++;
        }

        //write cached_pq_buff_to_write into file
        outFile.write(reinterpret_cast<const char*>(cached_pq_buff_to_write.get()), cached_PQ_nodes.size() * safe_num_pq_chunk);
        //write nodeID_pqIdx_map into file
        outFile.write(reinterpret_cast<const char*>(nodeID_pqIdx_map.data()), nodeID_pqIdx_map.size() * sizeof(uint32_t));
    }
    //no extra data structure in addition to the pq data
    else{
        //just write this reorder_pq_data_buff into the data file; make sure in the order of new_node_ID
        outFile.write(reinterpret_cast<const char*>(reorder_pq_data_buff.get()), points_num * safe_num_pq_chunk);
    }

    outFile.close();

    std::cout << "Final number of cached PQ: " << num_cached_nodes << "\n";
    std::cout << std::fixed << std::setprecision(4) << "Final actual cached PQ ratio in Mem: " << static_cast<double>(num_cached_nodes) / points_num << "\n";
    std::cout << std::fixed << std::setprecision(4) << "Expected cached PQ ratio: " << target_pq_cache_ratio << "\n";
    
    return size_cachedNodeID_PQidx_map + size_final_cached_PQ;
}

template <typename T>
/**
 * @brief Build a page-level graph index from a Vamana vector-level index.
 *
 * This function transforms a traditional Vamana graph (where each node is a single vector)
 * into PageANN's page-level graph structure (where multiple vectors are co-located in pages).
 * The transformation process includes:
 * 1. Validating input files (Vamana index, PQ data, pivots)
 * 2. Computing optimal page parameters based on memory budget
 * 3. Merging nearby vectors into pages using graph topology
 * 4. Creating a new disk layout optimized for sequential page access
 * 5. Reorganizing PQ data to match the new page structure
 *
 * @tparam T Data type of vectors (float, int8_t, or uint8_t)
 * @param index_prefix_path Path prefix for input Vamana index files
 * @param data_file_to_use Path to raw vector dataset
 * @param min_degree_per_node Minimum average degree per vector in page graph
 * @param R Maximum degree of input Vamana graph
 * @param num_pq_chunks_32 Number of PQ chunks for compression
 * @param compareMetric Distance metric (L2, INNER_PRODUCT, or COSINE)
 * @param memBudgetInGB Memory budget in GB for search process (affects PQ caching)
 * @param full_ooc If true, use fully out-of-core mode (no caching)
 * @return 0 on success, non-zero on failure
 */
int build_page_graph(const std::string &index_prefix_path, const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t R, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, float memBudgetInGB, bool full_ooc, const std::string &new_to_old_ids_map_file, bool enable_spare_fill)
{
    size_t points_num, dim;
    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

    // Step 1: Validate that all required input files exist and can be opened
    std::string index_file = index_prefix_path + "_disk.index";
    const std::string pq_compressed_all_nodes_path = index_prefix_path + "_pq_compressed.bin";
    const std::string pq_pivot_file = index_prefix_path + "_pq_pivots.bin";
    const std::string sample_data_path = index_prefix_path + "_sample_data.bin";
    const std::string sample_ids_path = index_prefix_path + "_sample_ids.bin";

    std::ifstream diskANN_index_ifs(index_file, std::ios::binary);
    if (!diskANN_index_ifs.is_open()) {
        throw std::runtime_error("Failed to open DiskANN index file: " + index_file);
    }
    diskANN_index_ifs.close();

    std::ifstream pq_data_ifs(pq_compressed_all_nodes_path, std::ios::binary);
    if (!pq_data_ifs.is_open()) {
        throw std::runtime_error("Failed to open PQ compressed data file: " + pq_compressed_all_nodes_path);
    }
    pq_data_ifs.close();

    std::ifstream pq_pivots_ifs(pq_pivot_file, std::ios::binary);
    if (!pq_pivots_ifs.is_open()) {
        throw std::runtime_error("Failed to open PQ pivots file: " + pq_pivot_file);
    }
    pq_pivots_ifs.close();

    // Step 2: Calculate memory allocation ratios for LSH and PQ caching
    // based on the available memory budget
    float pq_cache_ratio = getCacheRatio(memBudgetInGB, static_cast<size_t>(num_pq_chunks_32), points_num);

    if (pq_cache_ratio < 0){
        std::cout << "Error: memory budget shoud be at least 0.07 GB. Please increase your memory budget." << std::endl;
        return 0;
    }

    // Override caching ratios if fully out-of-core mode is enabled
    if(full_ooc){
        pq_cache_ratio = 0.0;
    }

    // Step 3: Calculate the maximum possible degree per page based on sector size constraints
    // Each neighbor can be either cached (4 bytes for ID only) or uncached (4 bytes + PQ data)
    double cached_size = sizeof(uint32_t);
    double uncached_size = sizeof(uint32_t) + static_cast<double>(num_pq_chunks_32);
    double avg_size_per_neighbor = pq_cache_ratio * cached_size + (1.0 - pq_cache_ratio) * uncached_size;
    double available_bytes = static_cast<double>(defaults::SECTOR_LEN) - static_cast<double>(dim * sizeof(T)) - static_cast<double>(sizeof(uint16_t) * 2); // Two uint16 metadata fields for neighbor counts
    size_t max_degree = static_cast<size_t>(std::floor(available_bytes / avg_size_per_neighbor));

    // Validate that the requested minimum degree is achievable within sector size limits
    if (min_degree_per_node > max_degree){
        std::cout << "Error: a page cannot host a single vector with the given degree. Please decrease your min degree requirement for each node." << std::endl;
        return 0;
    }

    // Step 4: Compute optimal page parameters (degree and vectors per page)
    // that maximize space utilization while meeting min_degree_per_node constraint
    auto result2 = get_optimal_pageann_degree_megaNode_capacity(dim, sizeof(T), static_cast<size_t>(num_pq_chunks_32), pq_cache_ratio, static_cast<size_t>(min_degree_per_node));
    uint32_t page_graph_degree = static_cast<uint32_t>(result2.first);
    uint32_t megaNode_capacity = static_cast<uint32_t>(result2.second);

    if (megaNode_capacity <= 0){
        std::cout << "Error: a page cannot host a single vector with the given degree. Please decrease your min degree requirement for each node." << std::endl;
        return 0;
    }

    std::cout << "Page-node graph max degree: " << page_graph_degree << std::endl;

    // Step 5: Initialize data and graph stores for loading input Vamana index
    std::shared_ptr<InMemOOCDataStore<T>> data_store = IndexFactory::construct_ooc_datastore<T>(DataStoreStrategy::MEMORY, points_num, dim, compareMetric);
    size_t vamana_degree = static_cast<size_t>(R);
    std::shared_ptr<InMemOOCGraphStore> graph_store = IndexFactory::construct_ooc_graphstore(GraphStoreStrategy::MEMORY, points_num, vamana_degree);

    // Step 6: Load raw vector data and Vamana graph topology from disk
    data_store->load(data_file_to_use);
    graph_store->set_type_size(sizeof(T));
    graph_store->load(index_file, points_num);

    // Step 7: Merge vectors into pages using graph-aware clustering
    // This groups vectors that are likely to be accessed together during search
    std::vector<std::vector<uint32_t>> mergedNodes;  // mergedNodes[pageID] = list of vector IDs in that page
    std::vector<uint32_t> nodeToMegaNodeMap;              // nodeToMegaNodeMap[vectorID] = pageID
    std::vector<uint32_t> new_to_original_map;        // Maps new ordering to original vector IDs
    const std::string final_index_prefix_path = index_prefix_path + "_PGD" + std::to_string(page_graph_degree) + "_PageANN";

    if (!new_to_old_ids_map_file.empty()) {
        // Load pre-computed node ordering from file, skip mergeNodesIntoMegaNodes
        if (!std::filesystem::exists(new_to_old_ids_map_file)) {
            std::cerr << "Error: new_to_old_ids_map_file not found: " << new_to_old_ids_map_file << std::endl;
            return -1;
        }
        diskann::cout << "Loading pre-computed node ordering from: " << new_to_old_ids_map_file << std::endl;
        new_to_original_map = diskann::loadTags(new_to_old_ids_map_file, data_file_to_use);
        if (new_to_original_map.size() != points_num) {
            std::cerr << "Error: new_to_old_ids_map size (" << new_to_original_map.size()
                      << ") != points_num (" << points_num << "). Aborting." << std::endl;
            return -1;
        }
        const uint64_t expected_num_pages = (points_num + megaNode_capacity - 1) / megaNode_capacity;
        mergedNodes.reserve(expected_num_pages);
        nodeToMegaNodeMap.assign(points_num, std::numeric_limits<uint32_t>::max());
        for (uint32_t pid = 0; pid < (uint32_t)expected_num_pages; pid++) {
            std::vector<uint32_t> groupedNodes;
            groupedNodes.reserve(megaNode_capacity);
            for (uint32_t k = 0; k < (uint32_t)megaNode_capacity; k++) {
                uint32_t new_nid = pid * (uint32_t)megaNode_capacity + k;
                uint32_t old_nid = new_to_original_map[new_nid];
                nodeToMegaNodeMap[old_nid] = pid;
                groupedNodes.push_back(old_nid);
            }
            mergedNodes.push_back(std::move(groupedNodes));
        }
        diskann::cout << "Loaded " << new_to_original_map.size() << " node mappings, "
                      << mergedNodes.size() << " pages." << std::endl;
    } else {
        mergeNodesIntoMegaNodes((uint64_t)megaNode_capacity, graph_store, data_store, (uint32_t)points_num, (uint32_t)dim, final_index_prefix_path, R, num_pq_chunks_32, mergedNodes, nodeToMegaNodeMap, new_to_original_map);
    }
    
    // Step 8: Create disk layout and write page-based index to disk
    const std::string final_outputFile = final_index_prefix_path + ".index";
    const std::string pq_compressed_reorder_path = final_index_prefix_path + "_reorder_pq_compressed.bin";
    const std::string pq_pivot_new_name_file = final_index_prefix_path + "_pq_pivots.bin";
    std::filesystem::copy_file(pq_pivot_file, pq_pivot_new_name_file, std::filesystem::copy_options::overwrite_existing);

    std::unique_ptr<uint8_t[]> reorder_pq_data_buff;  // Buffer for reordered PQ data
    tsl::robin_set<uint32_t> cached_PQ_nodes;         // Set of node IDs whose PQ data will be cached
    float scale = 1.0f;
    float scaled_pq_cache_ratio = scale * pq_cache_ratio;
    std::cout << std::fixed << std::setprecision(4) << "Scaled PQ cache ratio: " << scaled_pq_cache_ratio << std::endl;
    create_pageann_disk_layout(reorder_pq_data_buff, cached_PQ_nodes, (uint64_t)megaNode_capacity, scaled_pq_cache_ratio, graph_store, data_store, dim, page_graph_degree, pq_compressed_all_nodes_path, final_outputFile, mergedNodes, nodeToMegaNodeMap, 250, enable_spare_fill);

    // Step 9: Generate reorganized PQ compressed file matching new page layout
    size_t pq_mem_use_in_byte = generate_pageann_pq_data(pq_cache_ratio, points_num, reorder_pq_data_buff, cached_PQ_nodes, num_pq_chunks_32, pq_compressed_reorder_path);

    // Step 10: Calculate memory usage and provide LSH recommendations
    // Note: LSH (Locality-Sensitive Hashing) is currently not supported unless all PQ data fits in memory
    // LSH generation should be done separately if needed

    float actual_mem_usage_in_MB = static_cast<float>(pq_mem_use_in_byte) / (1024.0f * 1024.0f);
    std::cout << std::fixed << std::setprecision(4) << "Actual mem usage without in-memory index: " << actual_mem_usage_in_MB << " MB." << std::endl;

    // Step 11: copy sample data files to PageANN prefix if they exist
    // (used later by search binary for frequency-based cache ordering; not required for index build)
    const std::string pageann_sample_data_path = final_index_prefix_path + "_sample_data.bin";
    const std::string pageann_sample_ids_path = final_index_prefix_path + "_sample_ids.bin";
    if (std::filesystem::exists(sample_data_path) && std::filesystem::exists(sample_ids_path)) {
        std::filesystem::copy_file(sample_data_path, pageann_sample_data_path, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(sample_ids_path, pageann_sample_ids_path, std::filesystem::copy_options::overwrite_existing);
    } else {
        std::cout << "Note: sample_data.bin not found at vamana prefix, skipping copy. "
                  << "Generate it before search if frequency-based caching is needed." << std::endl;
    }

    return 0;  // Success
}



/**
 * @brief Calculate optimal PQ cache ratio based on memory budget.
 * This function determines how to allocate the available memory budget between:
 * - PQ caching: Storing compressed vector data in RAM for faster distance computations
 * - LSH sampling: Building a hash-based routing index for faster entry point selection
 *
 * Priority is given to PQ caching. If the budget allows caching all PQ data, remaining
 * memory is allocated for LSH. Otherwise, it incrementally increases PQ cache ratio
 * until the budget is exhausted.
 *
 * @param memBudgetInGB Available memory budget in GB for search process
 * @param PQ_size Number of bytes per PQ-compressed vector
 * @param npts Total number of vectors in the dataset
 * @return  pq_cache_ratio
 */
float getCacheRatio(float memBudgetInGB, size_t PQ_size, size_t npts) {
    const int INT32_SIZE = 4;
    const int MIN_MEM_MB = 60; //this is based on our experimental result
    float memBudgetInByte = (memBudgetInGB * 1024 - MIN_MEM_MB) * 1024 * 1024;

    if (memBudgetInByte < 0) {
        std::cout << "Memory budget should be at least " << (MIN_MEM_MB / 1024.0 + 0.001) << " GB\n";//to be accurate after three points after decimal
        return -1.0f;
    }

    size_t total_PQ_size = PQ_size * npts;
    float pq_cache_ratio = 0;
    float max_mem_usage = 0;

    float pre_pq_cache_ratio = 0;
    float pre_max_mem_usage = 0;
    std::cout << "Total number of points: " << npts << "\n";
    std::cout << "Actual spare memory budget: " << (memBudgetInGB * 1024 - MIN_MEM_MB) << " MB\n";

    //give the highest priority to pq:
    if(memBudgetInByte >= total_PQ_size){
        std::cout << "Memory budget is enough to load all PQ. So we prioritize storing all PQ in RAM." << std::endl;
        return 1.0;
    }

    const float step = 0.001;
    int i = 0;
    while (max_mem_usage <= memBudgetInByte) {
        pre_max_mem_usage = max_mem_usage;
        pre_pq_cache_ratio = pq_cache_ratio;

        pq_cache_ratio = i * step;

        if (i > 1000) {
            //std::cout << "pq_cache_ratio: " << pq_cache_ratio << " exceed the max sample ratio\n";
            break;
        }

        size_t cached_npts = static_cast<size_t>(pq_cache_ratio * npts);
        size_t cached_PQ_size = PQ_size * cached_npts;
        size_t uncached_PQ_size = total_PQ_size - cached_PQ_size;

        size_t size_cachedNodeID_PQidx_map = 0;
        if (pq_cache_ratio >= 0.25) {
            size_cachedNodeID_PQidx_map = INT32_SIZE * npts;
        } else {
            size_cachedNodeID_PQidx_map = 2 * INT32_SIZE * cached_npts * 2;//because sparse_map has a load factor of 2
        }

        size_t size_final_cached_PQ = 0;
        if (uncached_PQ_size <= size_cachedNodeID_PQidx_map) {
            size_final_cached_PQ = total_PQ_size;
            size_cachedNodeID_PQidx_map = 0;
            pq_cache_ratio = 1.0;
        } else {
            size_final_cached_PQ = cached_PQ_size;
        }

        max_mem_usage = static_cast<float>(size_final_cached_PQ + size_cachedNodeID_PQidx_map);

        i++;
    }

    std::cout << "Final max_mem_usage: " << pre_max_mem_usage / (1024 * 1024)
              << " MB, pq_cache_ratio: " << pre_pq_cache_ratio << "\n";

    return pre_pq_cache_ratio;
}


/***************************************************
    Logic of diskann graph construction starts here
 ***************************************************/

template <typename T, typename LabelT>
int build_disk_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                     diskann::Metric compareMetric, bool use_opq, const std::string &codebook_prefix, bool use_filters,
                     const std::string &label_file, const std::string &universal_label, const uint32_t filter_threshold,
                     const uint32_t Lf)
{
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
    {
        param_list.push_back(cur_param);
    }
    if (param_list.size() < 5 || param_list.size() > 9)
    {
        diskann::cout << "Correct usage of parameters is R (max degree)\n"
                         "L (indexing list size, better if >= R)\n"
                         "B (RAM limit of final index in GB)\n"
                         "M (memory limit while indexing)\n"
                         "T (number of threads for indexing)\n"
                         "B' (PQ bytes for disk index: optional parameter for "
                         "very large dimensional data)\n"
                         "reorder (set true to include full precision in data file"
                         ": optional paramter, use only when using disk PQ\n"
                         "build_PQ_byte (number of PQ bytes for inde build; set 0 to use "
                         "full precision vectors)\n"
                         "QD Quantized Dimension to overwrite the derived dim from B "
                      << std::endl;
        return -1;
    }

    if (!std::is_same<T, float>::value &&
        (compareMetric == diskann::Metric::INNER_PRODUCT || compareMetric == diskann::Metric::COSINE))
    {
        std::stringstream stream;
        stream << "Disk-index build currently only supports floating point data for Max "
                  "Inner Product Search/ cosine similarity. "
               << std::endl;
        throw diskann::ANNException(stream.str(), -1);
    }

    size_t disk_pq_dims = 0;
    bool use_disk_pq = false;
    size_t build_pq_bytes = 0;

    // if there is a 6th parameter, it means we compress the disk index
    // vectors also using PQ data (for very large dimensionality data). If the
    // provided parameter is 0, it means we store full vectors.
    if (param_list.size() > 5)
    {
        disk_pq_dims = atoi(param_list[5].c_str());
        use_disk_pq = true;
        if (disk_pq_dims == 0)
            use_disk_pq = false;
    }

    bool reorder_data = false;
    if (param_list.size() >= 7)
    {
        if (1 == atoi(param_list[6].c_str()))
        {
            reorder_data = true;
        }
    }

    if (param_list.size() >= 8)
    {
        build_pq_bytes = atoi(param_list[7].c_str());
    }

    std::string base_file(dataFilePath);
    std::string data_file_to_use = base_file;
    std::string labels_file_original = label_file;
    std::string index_prefix_path(indexFilePath);
    std::string labels_file_to_use = index_prefix_path + "_label_formatted.txt";
    std::string pq_pivots_path_base = codebook_prefix;
    std::string pq_pivots_path = file_exists(pq_pivots_path_base) ? pq_pivots_path_base + "_pq_pivots.bin"
                                                                  : index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";

    std::string labels_to_medoids_path = disk_index_path + "_labels_to_medoids.txt";
    std::string mem_labels_file = mem_index_path + "_labels.txt";
    std::string disk_labels_file = disk_index_path + "_labels.txt";
    std::string mem_univ_label_file = mem_index_path + "_universal_label.txt";
    std::string disk_univ_label_file = disk_index_path + "_universal_label.txt";
    std::string disk_labels_int_map_file = disk_index_path + "_labels_map.txt";
    std::string dummy_remap_file = disk_index_path + "_dummy_map.txt"; // remap will be used if we break-up points of
                                                                       // high label-density to create copies

    std::string sample_base_prefix = index_prefix_path + "_sample";
    // optional, used if disk index file must store pq data
    std::string disk_pq_pivots_path = index_prefix_path + "_disk.index_pq_pivots.bin";
    // optional, used if disk index must store pq data
    std::string disk_pq_compressed_vectors_path = index_prefix_path + "_disk.index_pq_compressed.bin";
    std::string prepped_base =
        index_prefix_path +
        "_prepped_base.bin"; // temp file for storing pre-processed base file for cosine/ mips metrics
    bool created_temp_file_for_processed_data = false;

    // output a new base file which contains extra dimension with sqrt(1 -
    // ||x||^2/M^2) for every x, M is max norm of all points. Extra space on
    // disk needed!
    if (compareMetric == diskann::Metric::INNER_PRODUCT)
    {
        Timer timer;
        std::cout << "Using Inner Product search, so need to pre-process base "
                     "data into temp file. Please ensure there is additional "
                     "(n*(d+1)*4) bytes for storing pre-processed base vectors, "
                     "apart from the interim indices created by DiskANN and the final index."
                  << std::endl;
        data_file_to_use = prepped_base;
        float max_norm_of_base = diskann::prepare_base_for_inner_products<T>(base_file, prepped_base);
        std::string norm_file = disk_index_path + "_max_base_norm.bin";
        diskann::save_bin<float>(norm_file, &max_norm_of_base, 1, 1);
        diskann::cout << timer.elapsed_seconds_for_step("preprocessing data for inner product") << std::endl;
        created_temp_file_for_processed_data = true;
    }
    else if (compareMetric == diskann::Metric::COSINE)
    {
        Timer timer;
        std::cout << "Normalizing data for cosine to temporary file, please ensure there is additional "
                     "(n*d*4) bytes for storing normalized base vectors, "
                     "apart from the interim indices created by DiskANN and the final index."
                  << std::endl;
        data_file_to_use = prepped_base;
        diskann::normalize_data_file(base_file, prepped_base);
        diskann::cout << timer.elapsed_seconds_for_step("preprocessing data for cosine") << std::endl;
        created_temp_file_for_processed_data = true;
    }

    uint32_t R = (uint32_t)atoi(param_list[0].c_str());
    uint32_t L = (uint32_t)atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0)
    {
        std::cerr << "Insufficient memory budget (or string was not in right "
                     "format). Should be > 0."
                  << std::endl;
        return -1;
    }
    double indexing_ram_budget = (float)atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0)
    {
        std::cerr << "Not building index. Please provide more RAM budget" << std::endl;
        return -1;
    }
    uint32_t num_threads = (uint32_t)atoi(param_list[4].c_str());

    if (num_threads != 0)
    {
        omp_set_num_threads(num_threads);
        mkl_set_num_threads(num_threads);
    }

    diskann::cout << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
                  << " Indexing ram budget: " << indexing_ram_budget << " T: " << num_threads << " Use_disk_pq: "<< use_disk_pq << std::endl;

    auto s = std::chrono::high_resolution_clock::now();

    // If there is filter support, we break-up points which have too many labels
    // into replica dummy points which evenly distribute the filters. The rest
    // of index build happens on the augmented base and labels
    std::string augmented_data_file, augmented_labels_file;
    if (use_filters)
    {
        convert_labels_string_to_int(labels_file_original, labels_file_to_use, disk_labels_int_map_file,
                                     universal_label);
        augmented_data_file = index_prefix_path + "_augmented_data.bin";
        augmented_labels_file = index_prefix_path + "_augmented_labels.txt";
        if (filter_threshold != 0)
        {
            breakup_dense_points<T>(data_file_to_use, labels_file_to_use, filter_threshold, augmented_data_file,
                                    augmented_labels_file,
                                    dummy_remap_file); // RKNOTE: This has large memory footprint,
                                                       // need to make this streaming
            data_file_to_use = augmented_data_file;
            labels_file_to_use = augmented_labels_file;
        }
    }

    size_t points_num, dim;

    Timer timer;
    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);
    const double p_val = ((double)MAX_PQ_TRAINING_SET_SIZE / (double)points_num);

    if (use_disk_pq)
    {
        generate_disk_quantized_data<T>(data_file_to_use, disk_pq_pivots_path, disk_pq_compressed_vectors_path,
                                        compareMetric, p_val, disk_pq_dims);
    }
    size_t num_pq_chunks = (size_t)(std::floor)(uint64_t(final_index_ram_limit / points_num));

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    if (param_list.size() >= 9 && atoi(param_list[8].c_str()) <= MAX_PQ_CHUNKS && atoi(param_list[8].c_str()) > 0)
    {
        std::cout << "Use quantized dimension (QD) to overwrite derived quantized "
                     "dimension from search_DRAM_budget (B)"
                  << std::endl;
        num_pq_chunks = atoi(param_list[8].c_str());
    }

    diskann::cout << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector."
                  << std::endl;

    generate_quantized_data<T>(data_file_to_use, pq_pivots_path, pq_compressed_vectors_path, compareMetric, p_val,
                               num_pq_chunks, use_opq, codebook_prefix);
    diskann::cout << timer.elapsed_seconds_for_step("generating quantized data") << std::endl;

// Gopal. Splitting diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
    MallocExtension::instance()->ReleaseFreeMemory();
#endif
    // Whether it is cosine or inner product, we still L2 metric due to the pre-processing.
    timer.reset();
    diskann::build_merged_vamana_index<T, LabelT>(data_file_to_use.c_str(), diskann::Metric::L2, L, R, p_val,
                                                  indexing_ram_budget, mem_index_path, medoids_path, centroids_path,
                                                  build_pq_bytes, use_opq, num_threads, use_filters, labels_file_to_use,
                                                  labels_to_medoids_path, universal_label, Lf);
    diskann::cout << timer.elapsed_seconds_for_step("building merged vamana index") << std::endl;

    timer.reset();
    if (!use_disk_pq)
    {
        diskann::diskann_create_disk_layout<T>(data_file_to_use.c_str(), mem_index_path, disk_index_path);
    }
    else
    {
        if (!reorder_data)
            diskann::diskann_create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path, mem_index_path, disk_index_path);
        else
            diskann::diskann_create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path, mem_index_path, disk_index_path,
                                                 data_file_to_use.c_str());
    }
    diskann::cout << timer.elapsed_seconds_for_step("generating disk layout") << std::endl;

    double ten_percent_points = std::ceil(points_num * 0.1);
    double num_sample_points =
        ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : ten_percent_points;
    double sample_sampling_rate = num_sample_points / points_num;
    gen_random_slice<T>(data_file_to_use.c_str(), sample_base_prefix, sample_sampling_rate);
    if (use_filters)
    {
        copy_file(labels_file_to_use, disk_labels_file);
        std::remove(mem_labels_file.c_str());
        if (universal_label != "")
        {
            copy_file(mem_univ_label_file, disk_univ_label_file);
            std::remove(mem_univ_label_file.c_str());
        }
        std::remove(augmented_data_file.c_str());
        std::remove(augmented_labels_file.c_str());
        std::remove(labels_file_to_use.c_str());
    }
    if (created_temp_file_for_processed_data)
        std::remove(prepped_base.c_str());
    std::remove(mem_index_path.c_str());
    std::remove((mem_index_path + ".data").c_str());
    std::remove((mem_index_path + ".tags").c_str());
    if (use_disk_pq)
        std::remove(disk_pq_compressed_vectors_path.c_str());

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    diskann::cout << "Indexing time: " << diff.count() << std::endl;

    return 0;
}

template <typename T>
void diskann_create_disk_layout(const std::string base_file, const std::string mem_index_file, const std::string output_file,
                        const std::string reorder_data_file)
{
    uint32_t npts, ndims;

    // amount to read or write in one shot
    size_t read_blk_size = 64 * 1024 * 1024;
    size_t write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *)&npts, sizeof(uint32_t));
    base_reader.read((char *)&ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // Check if we need to append data for re-ordering
    bool append_reorder_data = false;
    std::ifstream reorder_data_reader;

    uint32_t npts_reorder_file = 0, ndims_reorder_file = 0;
    if (reorder_data_file != std::string(""))
    {
        append_reorder_data = true;
        size_t reorder_data_file_size = get_file_size(reorder_data_file);
        reorder_data_reader.exceptions(std::ofstream::failbit | std::ofstream::badbit);

        try
        {
            reorder_data_reader.open(reorder_data_file, std::ios::binary);
            reorder_data_reader.read((char *)&npts_reorder_file, sizeof(uint32_t));
            reorder_data_reader.read((char *)&ndims_reorder_file, sizeof(uint32_t));
            if (npts_reorder_file != npts)
                throw ANNException("Mismatch in num_points between reorder "
                                   "data file and base file",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            if (reorder_data_file_size != 8 + sizeof(float) * (size_t)npts_reorder_file * (size_t)ndims_reorder_file)
                throw ANNException("Discrepancy in reorder data file size ", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        catch (std::system_error &e)
        {
            throw FileException(reorder_data_file, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    // create cached reader + writer
    size_t actual_file_size = get_file_size(mem_index_file);
    diskann::cout << "Vamana index file size=" << actual_file_size << std::endl;
    std::ifstream vamana_reader(mem_index_file, std::ios::binary);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    uint32_t width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *)&index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
                  "meta-data."
               << " file size from file: " << index_file_size << " actual file size: " << actual_file_size << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;

    vamana_reader.read((char *)&width_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&medoid_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&vamana_frozen_num, sizeof(uint64_t));
    // compute
    uint64_t medoid, max_node_len, nnodes_per_sector;
    npts_64 = (uint64_t)npts;
    medoid = (uint64_t)medoid_u32;
    if (vamana_frozen_num == 1)
        vamana_frozen_loc = medoid;
    max_node_len = (((uint64_t)width_u32 + 1) * sizeof(uint32_t)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = defaults::SECTOR_LEN / max_node_len; // 0 if max_node_len > SECTOR_LEN

    diskann::cout << "medoid: " << medoid << "B" << std::endl;
    diskann::cout << "max_node_len: " << max_node_len << "B" << std::endl;
    diskann::cout << "nnodes_per_sector: " << nnodes_per_sector << "B" << std::endl;

    // defaults::SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, defaults::SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    uint32_t &nnbrs = *(uint32_t *)(node_buf.get() + ndims_64 * sizeof(T));
    uint32_t *nhood_buf = (uint32_t *)(node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(uint32_t));

    // number of sectors (1 for meta data)
    uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                               : npts_64 * DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
    uint64_t n_reorder_sectors = 0;
    uint64_t n_data_nodes_per_sector = 0;

    if (append_reorder_data)
    {
        n_data_nodes_per_sector = defaults::SECTOR_LEN / (ndims_reorder_file * sizeof(float));
        n_reorder_sectors = ROUND_UP(npts_64, n_data_nodes_per_sector) / n_data_nodes_per_sector;
    }
    uint64_t disk_index_file_size = (n_sectors + n_reorder_sectors + 1) * defaults::SECTOR_LEN;

    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back((uint64_t)append_reorder_data);
    if (append_reorder_data)
    {
        output_file_meta.push_back(n_sectors + 1);
        output_file_meta.push_back(ndims_reorder_file);
        output_file_meta.push_back(n_data_nodes_per_sector);
    }
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    diskann::cout << "# sectors: " << n_sectors << std::endl;
    uint64_t cur_node_id = 0;

    if (nnodes_per_sector > 0)
    { // Write multiple nodes per sector
        for (uint64_t sector = 0; sector < n_sectors; sector++)
        {
            if (sector % 100000 == 0)
            {
                diskann::cout << "Sector #" << sector << "written" << std::endl;
            }
            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
            for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
                 sector_node_id++)
            {
                memset(node_buf.get(), 0, max_node_len);
                // read cur node's nnbrs
                vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

                // sanity checks on nnbrs
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);

                // read node's nhood
                vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                if (nnbrs > width_u32)
                {
                    vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                }

                // write coords of node first
                //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
                memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

                // write nnbrs
                *(uint32_t *)(node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

                // write nhood next
                memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                       (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

                // get offset into sector_buf
                char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

                // copy node buf into sector_node_buf
                memcpy(sector_node_buf, node_buf.get(), max_node_len);
                cur_node_id++;
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
        }
    }
    else
    { // Write multi-sector nodes
        uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
        for (uint64_t i = 0; i < npts_64; i++)
        {
            if ((i * nsectors_per_node) % 100000 == 0)
            {
                diskann::cout << "Sector #" << i * nsectors_per_node << "written" << std::endl;
            }
            memset(multisector_buf.get(), 0, nsectors_per_node * defaults::SECTOR_LEN);

            memset(node_buf.get(), 0, max_node_len);
            // read cur node's nnbrs
            vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

            // sanity checks on nnbrs
            assert(nnbrs > 0);
            assert(nnbrs <= width_u32);

            // read node's nhood
            vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
            if (nnbrs > width_u32)
            {
                vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
            }

            // write coords of node first
            //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
            base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
            memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

            // write nnbrs
            *(uint32_t *)(multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

            // write nhood next
            memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                   (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

            // flush sector to disk
            diskann_writer.write(multisector_buf.get(), nsectors_per_node * defaults::SECTOR_LEN);
        }
    }

    if (append_reorder_data)
    {
        diskann::cout << "Index written. Appending reorder data..." << std::endl;

        auto vec_len = ndims_reorder_file * sizeof(float);
        std::unique_ptr<char[]> vec_buf = std::make_unique<char[]>(vec_len);

        for (uint64_t sector = 0; sector < n_reorder_sectors; sector++)
        {
            if (sector % 100000 == 0)
            {
                diskann::cout << "Reorder data Sector #" << sector << "written" << std::endl;
            }

            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);

            for (uint64_t sector_node_id = 0; sector_node_id < n_data_nodes_per_sector && sector_node_id < npts_64;
                 sector_node_id++)
            {
                memset(vec_buf.get(), 0, vec_len);
                reorder_data_reader.read(vec_buf.get(), vec_len);

                // copy node buf into sector_node_buf
                memcpy(sector_buf.get() + (sector_node_id * vec_len), vec_buf.get(), vec_len);
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
        }
    }
    diskann_writer.close();
    diskann::save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    diskann::cout << "Output disk index file written to " << output_file << std::endl;
}




template <typename T>
void create_laann_disk_layout(const uint64_t num_vectors_per_sector,
                       std::shared_ptr<InMemOOCGraphStore> graph_store,
                       std::shared_ptr<InMemOOCDataStore<T>> data_store, const size_t ndims,
                       const uint32_t optimal_vector_degree, const uint32_t L,
                       const std::string &final_index_prefix_path,
                       const std::vector<std::vector<uint32_t>>& mergedNodes,
                       const std::vector<uint32_t>& nodeToMegaNodeMap,
                       const uint32_t vamana_R) {

    // Input validation
    if (!graph_store || !data_store) {
        throw std::invalid_argument("Null store pointers provided");
    }
    if (num_vectors_per_sector == 0) {
        throw std::invalid_argument("num_vectors_per_sector must be greater than 0");
    }
    if (mergedNodes.empty()) {
        throw std::invalid_argument("mergedNodes cannot be empty");
    }
    if (nodeToMegaNodeMap.empty()) {
        throw std::invalid_argument("nodeToMegaNodeMap cannot be empty");
    }

    // Step 1: calculate metadata
    const uint64_t each_node_space = static_cast<uint64_t>(ndims * sizeof(T));
    const uint32_t megaNode_capacity = static_cast<uint32_t>(num_vectors_per_sector);  // no centroid
    //const uint32_t mega_node_degree = optimal_vector_degree * megaNode_capacity;
    const uint32_t mega_node_degree = (defaults::SECTOR_LEN - megaNode_capacity * ndims * sizeof(T))/sizeof(uint32_t) - 1;
    const size_t num_node = nodeToMegaNodeMap.size();
    const uint32_t num_pages = static_cast<uint32_t>(mergedNodes.size());

    diskann::cout << "Section Length: " << defaults::SECTOR_LEN << "B" << std::endl;
    diskann::cout << "each_node_space: " << each_node_space << "B" << std::endl;
    diskann::cout << "megaNode_capacity: " << megaNode_capacity << std::endl;
    diskann::cout << "optimal_vector_degree: " << optimal_vector_degree << std::endl;
    diskann::cout << "mega_node_degree: " << mega_node_degree << std::endl;
    diskann::cout << "Number of nodes in graph: " << num_node << std::endl;

    const uint64_t medoid = static_cast<uint64_t>(nodeToMegaNodeMap[graph_store->get_start()]);
    diskann::cout << "medoid mega node ID: " << medoid << std::endl;

    // Step 2: write metadata sector (reserved, filled at the end)
    constexpr size_t WRITE_BLK_SIZE = 64 * 1024 * 1024; // 64MB buffer
    constexpr size_t LOG_INTERVAL = 10000;

    const std::string laann_disk_index_file = final_index_prefix_path + ".index";
    cached_ofstream laann_data_writer(laann_disk_index_file, WRITE_BLK_SIZE);

    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back(static_cast<uint64_t>(num_node));              // 0: npts (actual, not aligned)
    output_file_meta.push_back(static_cast<uint64_t>(ndims));                 // 1: ndims
    output_file_meta.push_back(medoid);                                       // 2: medoid
    output_file_meta.push_back(each_node_space);                              // 3: each_node_space = ndims * sizeof(T)
    output_file_meta.push_back(static_cast<uint64_t>(megaNode_capacity));     // 4: megaNode_capacity
    output_file_meta.push_back(static_cast<uint64_t>(num_pages));             // 5: num_pages
    output_file_meta.push_back(static_cast<uint64_t>(mega_node_degree));      // 6: mega_node_degree (real max neighbors per page)
    output_file_meta.push_back(static_cast<uint64_t>(vamana_R));              // 7: vamana_R

    std::vector<char> sector_buf(defaults::SECTOR_LEN, 0);
    laann_data_writer.write(sector_buf.data(), defaults::SECTOR_LEN);

    // Step 2.5: build old_to_new map
    std::vector<uint32_t> old_to_new(num_node, UINT32_MAX);
    for (uint32_t mega_node_id = 0; mega_node_id < mergedNodes.size(); ++mega_node_id) {
        const auto &groupedNodes = mergedNodes[mega_node_id];
        for (uint32_t k = 0; k < groupedNodes.size(); ++k) {
            const uint32_t new_id = mega_node_id * megaNode_capacity + k;
            if (new_id >= num_node) break;
            old_to_new[groupedNodes[k]] = new_id;
        }
    }

    // Step 3: write data pages
    const auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t maxDeg = 0, minDeg = std::numeric_limits<uint32_t>::max();
    uint64_t totalDeg = 0;

    std::vector<T> cur_vec(data_store->get_aligned_dim());

    for (uint32_t mega_node_id = 0; mega_node_id < mergedNodes.size(); ++mega_node_id) {
        std::fill(sector_buf.begin(), sector_buf.end(), 0);
        const auto& group = mergedNodes[mega_node_id];

        size_t written_size = 0;

        // Write all vector data
        for (uint32_t k = 0; k < group.size(); ++k) {
            data_store->get_vector(group[k], cur_vec.data());
            std::memcpy(sector_buf.data() + written_size, cur_vec.data(), each_node_space);
            written_size += each_node_space;
        }

        // Step A: merge all Vamana neighbors from every vector in the page
        // Filter same-page vectors, deduplicate, up to mega_node_degree
        tsl::robin_set<uint32_t> mega_node_inserted_nbrs;
        mega_node_inserted_nbrs.reserve(vamana_R * group.size());
        std::vector<uint32_t> page_nbrs;
        page_nbrs.reserve(mega_node_degree);

        for (uint32_t k = 0; k < group.size() && page_nbrs.size() < mega_node_degree; ++k) {
            for (const auto& nbr_old_id : graph_store->get_ooc_neighbours(group[k])) {
                if (nodeToMegaNodeMap[nbr_old_id] == mega_node_id) continue;
                if (mega_node_inserted_nbrs.insert(nbr_old_id).second) {
                    page_nbrs.push_back(old_to_new[nbr_old_id]);
                    if (page_nbrs.size() >= mega_node_degree) break;
                }
            }
        }

        // Step B: fill spare space via beam search if merged neighbors < mega_node_degree
        if (page_nbrs.size() < mega_node_degree) {
            tsl::robin_set<uint32_t> page_vector_set;
            for (uint32_t vid : group) page_vector_set.insert(vid);

            uint32_t spare = mega_node_degree - static_cast<uint32_t>(page_nbrs.size());
            std::vector<Neighbor> candidates;
            for (size_t vec_idx = 0; vec_idx < group.size() && spare > 0; vec_idx++) {
                search_for_candidates<T>(group[vec_idx], graph_store, data_store, L, candidates);
                for (const auto& cand : candidates) {
                    if (spare == 0) break;
                    if (page_vector_set.count(cand.id)) continue;
                    if (mega_node_inserted_nbrs.insert(cand.id).second) {
                        page_nbrs.push_back(old_to_new[cand.id]);
                        spare--;
                    }
                }
            }
        }

        // Neighbor data is always written at fixed offset (megaNode_capacity * each_node_space)
        // so the reader can locate it regardless of how many vectors the last page has.
        written_size = static_cast<size_t>(megaNode_capacity) * each_node_space;

        // Write total neighbor count then neighbor IDs
        uint32_t actual_num_nbrs = static_cast<uint32_t>(page_nbrs.size());
        std::memcpy(sector_buf.data() + written_size, &actual_num_nbrs, sizeof(uint32_t));
        written_size += sizeof(uint32_t);
        std::memcpy(sector_buf.data() + written_size, page_nbrs.data(), page_nbrs.size() * sizeof(uint32_t));

        laann_data_writer.write(sector_buf.data(), defaults::SECTOR_LEN);

        maxDeg = std::max(maxDeg, actual_num_nbrs);
        minDeg = std::min(minDeg, actual_num_nbrs);
        totalDeg += actual_num_nbrs;

        if (mega_node_id % LOG_INTERVAL == 0) {
            const float percent = 100.0f * (mega_node_id + 1) / mergedNodes.size();
            const auto now = std::chrono::high_resolution_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            diskann::cout << "\rData pages: "
                          << std::setw(6) << std::fixed << std::setprecision(2) << percent << "% ("
                          << mega_node_id + 1 << "/" << mergedNodes.size() << "), "
                          << "time: " << static_cast<int>(elapsed) << "s"
                          << std::flush;
        }
    }

    diskann::cout << "\nData pages written: " << mergedNodes.size() << std::endl;
    diskann::cout << "Max degree: " << maxDeg << std::endl;
    diskann::cout << "Min degree: " << minDeg << std::endl;
    diskann::cout << "Average degree: " << static_cast<double>(totalDeg) / mergedNodes.size() << std::endl;

    laann_data_writer.close();

    diskann::save_bin<uint64_t>(laann_disk_index_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    diskann::cout << "Output disk file written to " << laann_disk_index_file << std::endl;
}

size_t get_laann_megaNode_capacity(size_t dimension, size_t typeByte, size_t vectorDegree) {
    // Page layout: [capacity * dim * typeByte] [uint32_t: total_num_nbrs] [total_num_nbrs * uint32_t]
    // where total_num_nbrs = capacity * vectorDegree
    // Available bytes for vectors+neighbors = SECTOR_LEN - sizeof(uint32_t) (the count field)
    return (defaults::SECTOR_LEN - sizeof(uint32_t)) / (dimension * typeByte + sizeof(uint32_t) * vectorDegree);
}

std::pair<size_t, size_t> get_optimal_laann_degree_megaNode_capacity(size_t dimension, size_t typeByte, size_t min_degree_per_vector) {
    
    size_t vectorDegree = min_degree_per_vector;
    size_t target_vec_capacity_per_page = get_laann_megaNode_capacity(dimension, typeByte, vectorDegree);
    if (target_vec_capacity_per_page == 0){
        throw std::runtime_error("Error: a page cannot host a single vector with the given degree");
    }

    //find the optimal vectorDegree: maximize degree while capacity stays the same
    size_t maxDegree = 1023;
    while(vectorDegree < maxDegree && get_laann_megaNode_capacity(dimension, typeByte, vectorDegree + 1) == target_vec_capacity_per_page){
        vectorDegree++;
    }

    // Allow one extra degree if the next degree fits within 1% of the raw page neighbor budget.
    // The actual on-disk neighbor count is capped by mega_node_degree (raw budget) regardless,
    // so this only affects the check and the MGD value in the filename.
    size_t raw_total_budget = (defaults::SECTOR_LEN - target_vec_capacity_per_page * dimension * typeByte) / sizeof(uint32_t) - 1;
    if ((vectorDegree + 1) * target_vec_capacity_per_page * 99 <= raw_total_budget * 100) {
        vectorDegree++;
    }

    std::cout << "Final optimal choice: degree per vector: " << vectorDegree << ", target_vec_capacity_per_page: " << target_vec_capacity_per_page << ", max pageNode degree: " << vectorDegree * target_vec_capacity_per_page << "\n";

    return {vectorDegree, target_vec_capacity_per_page};
}

void generate_laann_pq_data(const size_t points_num, 
                         const uint32_t num_pq_chunks_32, 
                         const std::string &pq_compressed_reorder_path,
                         const std::string &pq_compressed_all_nodes_path,
                         const std::vector<uint32_t>& nodeToMegaNodeMap,
                         const std::vector<std::vector<uint32_t>>& mergedNodes,
                         const uint32_t megaNode_capacity) {
    
    //step 1: load pq data into RAM
    int all_pq_npts, pq_ndims;
    std::ifstream pqAllPointsReader;
    pqAllPointsReader.exceptions(std::ios::badbit | std::ios::failbit);
    pqAllPointsReader.open(pq_compressed_all_nodes_path, std::ios::binary);
    pqAllPointsReader.seekg(0, pqAllPointsReader.beg);
    pqAllPointsReader.read((char *)&all_pq_npts, sizeof(int));
    pqAllPointsReader.read((char *)&pq_ndims, sizeof(int));//i.e., num_pq_chunks
   
    if (static_cast<size_t>(all_pq_npts) != points_num) {
        throw std::runtime_error("PQ data points not match with original data points");
    }
    if (static_cast<uint32_t>(pq_ndims) != num_pq_chunks_32) {
        throw std::runtime_error("PQ dimensions mismatch: file=" + std::to_string(pq_ndims) + ", expected=" + std::to_string(num_pq_chunks_32));
    }
    diskann::cout << "Number of points of compressed PQ data: " << all_pq_npts << std::endl;
    diskann::cout << "Number of dimensions of compressed PQ data: " << pq_ndims << std::endl;

    size_t total_bytes = static_cast<size_t>(all_pq_npts) * pq_ndims;
    std::cout << "Total bytes to allocate: " << total_bytes << " (" << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << " GB)" << std::endl;
    std::unique_ptr<uint8_t[]> pq_compressed_page_all_points = std::make_unique<uint8_t[]>(total_bytes);
    pqAllPointsReader.read((char*)(pq_compressed_page_all_points.get()), total_bytes * sizeof(uint8_t));
    pqAllPointsReader.close();
    
    //step2: generate old to new map
    std::vector<uint32_t> old_to_new(points_num, UINT32_MAX);
    for (uint32_t mega_node_id = 0; mega_node_id < mergedNodes.size(); ++mega_node_id) {
        const auto &groupedNodes = mergedNodes[mega_node_id];
        for (uint32_t k = 0; k < groupedNodes.size(); ++k) {
            const uint32_t new_id = mega_node_id * megaNode_capacity + k; // Fixed: was megaNode_capacity
            if (new_id >= points_num) {
                break;
            }
            const uint32_t old_id = groupedNodes[k];
            old_to_new[old_id] = new_id;
        }
    }

    //write reorder pq data
    std::unique_ptr<uint8_t[]> reorder_pq_data_buff = std::make_unique<uint8_t[]>(total_bytes);
    ///MARK: we cannot use uint32_t to represent -- points_num * pq_ndims -- may overflow
    size_t safe_pq_ndims = static_cast<size_t>(pq_ndims);
    const size_t pq_size = safe_pq_ndims * sizeof(uint8_t);
    for (size_t old_node_ID = 0; old_node_ID < points_num; old_node_ID++){
        //this is only for dataset size <= 4B
        uint32_t new_reordered_node_ID = old_to_new[old_node_ID];
        memcpy(&reorder_pq_data_buff[new_reordered_node_ID * safe_pq_ndims], &pq_compressed_page_all_points[old_node_ID * safe_pq_ndims], pq_size);
    }

    ///now we begin to save PQ compress data into disk
    std::ofstream outFile(pq_compressed_reorder_path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file "<< pq_compressed_reorder_path << " for writing.\n";
    }
    //write metadata: number of total points, num_pq_chunks_32,
    uint32_t points_num_32 = static_cast<uint32_t>(points_num);
    outFile.write(reinterpret_cast<const char*>(&points_num_32), sizeof(points_num_32));
    outFile.write(reinterpret_cast<const char*>(&num_pq_chunks_32), sizeof(num_pq_chunks_32));

    //just write this reorder_pq_data_buff into the data file; make sure in the order of new_node_ID
    size_t safe_num_pq_chunk = static_cast<size_t>(num_pq_chunks_32);
    outFile.write(reinterpret_cast<const char*>(reorder_pq_data_buff.get()), points_num * safe_num_pq_chunk);
    outFile.close();

    std::cout << "Finish writing reordered PQ data.\n";
}

template <typename T>
int build_laann_graph(const std::string &index_prefix_path, const std::string &data_file_to_use, const uint32_t R, const uint32_t grouping_L, const uint32_t fill_L, const uint32_t min_degree_per_vector, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, bool use_greedy_grouping, const std::string &new_to_old_ids_map_file)
{
//0. preparation
    size_t points_num, dim;
    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

    //first, make sure all these files exist and can be opened
    std::string index_file = index_prefix_path + "_disk.index";
    const std::string pq_compressed_all_nodes_path = index_prefix_path + "_pq_compressed.bin";
    const std::string pq_pivot_file = index_prefix_path + "_pq_pivots.bin";
    const std::string sample_data_path = index_prefix_path + "_sample_data.bin";
    const std::string sample_ids_path = index_prefix_path + "_sample_ids.bin";

    std::ifstream diskANN_index_ifs(index_file, std::ios::binary);
    if (!diskANN_index_ifs.is_open()) {
        throw std::runtime_error("Failed to open DiskANN index file: " + index_file);
    }
    diskANN_index_ifs.close();

    std::ifstream pq_data_ifs(pq_compressed_all_nodes_path, std::ios::binary);
    if (!pq_data_ifs.is_open()) {
        throw std::runtime_error("Failed to open PQ compressed data file: " + pq_compressed_all_nodes_path);
    }
    pq_data_ifs.close();

    std::ifstream pq_pivots_ifs(pq_pivot_file, std::ios::binary);
    if (!pq_pivots_ifs.is_open()) {
        throw std::runtime_error("Failed to open PQ pivots file: " + pq_pivot_file);
    }
    pq_pivots_ifs.close();

//1. logic starts here
    auto ret = get_optimal_laann_degree_megaNode_capacity(dim, sizeof(T), static_cast<size_t>(min_degree_per_vector));
    uint32_t optimal_vector_degree = static_cast<uint32_t>(ret.first);
    uint32_t num_vectors_per_sector = static_cast<uint32_t>(ret.second);
    size_t vamana_degree = static_cast<size_t>(R);
    uint32_t megaNode_capacity = num_vectors_per_sector;  // no centroid
    uint32_t mega_node_degree = optimal_vector_degree * megaNode_capacity;

    std::cout << "# of vectors per page (megaNode_capacity) = " << megaNode_capacity << std::endl;
    std::cout << "degree per vector of base index = " << vamana_degree << std::endl;
    std::cout << "degree per vector in mega graph index = " << optimal_vector_degree << std::endl;
    std::cout << "degree per mega node = " << mega_node_degree << std::endl;
    

    if (num_vectors_per_sector <= 0){
        std::cout << "Error: a page cannot host a single vector." << std::endl;
        return 0;
    }

    std::shared_ptr<InMemOOCDataStore<T>> data_store = IndexFactory::construct_ooc_datastore<T>(DataStoreStrategy::MEMORY, points_num, dim, compareMetric);
    std::shared_ptr<InMemOOCGraphStore> graph_store = IndexFactory::construct_ooc_graphstore(GraphStoreStrategy::MEMORY, points_num, vamana_degree);   
    
    ///MARK: loading data and topology
    data_store->load(data_file_to_use);
    graph_store->set_type_size(sizeof(T));
    graph_store->load(index_file, points_num);

    ///MARK: verify uniform degree (required for LAANN disk format)
    uint32_t min_degree = graph_store->get_min_observed_degree();
    uint32_t max_degree = graph_store->get_max_observed_degree();

    std::cout << "Vamana graph degree: min=" << min_degree << ", max=" << max_degree << std::endl;

    ///MARK: merge nodes
    std::vector<std::vector<uint32_t>> mergedNodes;
    std::vector<uint32_t> nodeToMegaNodeMap;
    std::vector<uint32_t> new_to_original_map;
    const std::string final_index_prefix_path = index_prefix_path + "_MGD" + std::to_string(mega_node_degree) + "_LAANN";

    // Sanity check: R must not exceed optimal_vector_degree
    if (R > optimal_vector_degree) {
        std::cerr << "Error: R (" << R << ") > optimal_vector_degree (" << optimal_vector_degree
                  << "). Merged neighbor count could exceed page budget." << std::endl;
        return -1;
    }

    if (!new_to_old_ids_map_file.empty()) {
        // Load pre-computed node ordering from file, skip grouping
        if (!std::filesystem::exists(new_to_old_ids_map_file)) {
            std::cerr << "Error: new_to_old_ids_map_file not found: " << new_to_old_ids_map_file << std::endl;
            return -1;
        }
        diskann::cout << "Loading pre-computed node ordering from: " << new_to_old_ids_map_file << std::endl;
        new_to_original_map = diskann::loadTags(new_to_old_ids_map_file, data_file_to_use);
        if (new_to_original_map.size() != points_num) {
            std::cerr << "Error: new_to_old_ids_map size (" << new_to_original_map.size()
                      << ") != points_num (" << points_num << "). Aborting." << std::endl;
            return -1;
        }
        const uint64_t expected_num_pages = (points_num + megaNode_capacity - 1) / megaNode_capacity;
        mergedNodes.reserve(expected_num_pages);
        nodeToMegaNodeMap.assign(points_num, std::numeric_limits<uint32_t>::max());
        for (uint32_t pid = 0; pid < (uint32_t)expected_num_pages; pid++) {
            std::vector<uint32_t> groupedNodes;
            groupedNodes.reserve(megaNode_capacity);
            for (uint32_t k = 0; k < (uint32_t)megaNode_capacity; k++) {
                uint32_t new_nid = pid * (uint32_t)megaNode_capacity + k;
                uint32_t old_nid = new_to_original_map[new_nid];
                nodeToMegaNodeMap[old_nid] = pid;
                groupedNodes.push_back(old_nid);
            }
            mergedNodes.push_back(std::move(groupedNodes));
        }
        diskann::cout << "Loaded " << new_to_original_map.size() << " node mappings, "
                      << mergedNodes.size() << " pages." << std::endl;

        // Write id maps to the LAANN prefix so GT translation tools can use them
        {
            int npts_i32 = static_cast<int>(points_num);
            int dim_i32 = 1;
            // Compute original→new mapping (inverse of new_to_original_map)
            std::vector<uint32_t> original_to_new(points_num, UINT32_MAX);
            for (uint32_t new_id = 0; new_id < static_cast<uint32_t>(new_to_original_map.size()); new_id++) {
                uint32_t old_id = new_to_original_map[new_id];
                if (old_id < static_cast<uint32_t>(points_num))
                    original_to_new[old_id] = new_id;
            }
            const std::string old_to_new_map_file = final_index_prefix_path + "_original_to_new_ids_map.bin";
            std::ofstream ofs_o2n(old_to_new_map_file, std::ios::binary);
            ofs_o2n.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));
            ofs_o2n.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));
            ofs_o2n.write(reinterpret_cast<char*>(original_to_new.data()), original_to_new.size() * sizeof(uint32_t));
            ofs_o2n.close();
            diskann::cout << "Written original_to_new map to: " << old_to_new_map_file << std::endl;

            const std::string new_to_old_out = final_index_prefix_path + "_new_to_old_ids_map.bin";
            std::ofstream ofs_n2o(new_to_old_out, std::ios::binary);
            ofs_n2o.write(reinterpret_cast<char*>(&npts_i32), sizeof(int));
            ofs_n2o.write(reinterpret_cast<char*>(&dim_i32), sizeof(int));
            ofs_n2o.write(reinterpret_cast<char*>(new_to_original_map.data()), new_to_original_map.size() * sizeof(uint32_t));
            ofs_n2o.close();
            diskann::cout << "Written new_to_old map to: " << new_to_old_out << std::endl;
        }
    } else if (use_greedy_grouping) {
        if (grouping_L < megaNode_capacity) {
            std::cerr << "Error: grouping_L must be >= megaNode_capacity. grouping_L=" << grouping_L << ", megaNode_capacity=" << megaNode_capacity << std::endl;
            return -1;
        }
        std::cout << "Using greedy grouping (search-based), grouping_L=" << grouping_L << ", fill_L=" << fill_L << ", megaNode_capacity=" << megaNode_capacity << std::endl;
        mergeNodesIntoMegaNodesGreedy<T>((uint64_t)megaNode_capacity, graph_store, data_store,
            (uint32_t)points_num, final_index_prefix_path, grouping_L,
            mergedNodes, nodeToMegaNodeMap, new_to_original_map);
    } else {
        std::cout << "Using PageANN graph-topology merging (BFS), fill_L=" << fill_L << ", megaNode_capacity=" << megaNode_capacity << std::endl;
        mergeNodesIntoMegaNodes<T>((uint64_t)megaNode_capacity, graph_store, data_store,
            (uint32_t)points_num, (uint32_t)dim, final_index_prefix_path, R, num_pq_chunks_32,
            mergedNodes, nodeToMegaNodeMap, new_to_original_map);
    }

    std::cout << "Grouping complete. mergedNodes: " << mergedNodes.size() << " pages." << std::endl;

    ///MARK: write disk layout + fill spare space
    create_laann_disk_layout((uint64_t)num_vectors_per_sector, graph_store, data_store, dim, optimal_vector_degree, fill_L, final_index_prefix_path, mergedNodes, nodeToMegaNodeMap, R);
    
    ///MARK: generate new pq compressed file here
    const std::string pq_compressed_reorder_path = final_index_prefix_path + "_reorder_pq_compressed.bin";
    const std::string pq_pivot_new_name_file = final_index_prefix_path + "_pq_pivots.bin";
    std::filesystem::copy_file(pq_pivot_file, pq_pivot_new_name_file, std::filesystem::copy_options::overwrite_existing);
    generate_laann_pq_data(points_num, num_pq_chunks_32, pq_compressed_reorder_path, pq_compressed_all_nodes_path, nodeToMegaNodeMap, mergedNodes, megaNode_capacity);

    //rename the sample data generated by diskann
    const std::string meagann_sample_data_path = final_index_prefix_path + "_sample_data.bin";
    const std::string megaann_sample_ids_path = final_index_prefix_path + "_sample_ids.bin";
    std::filesystem::copy_file(sample_data_path, meagann_sample_data_path, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(sample_ids_path, megaann_sample_ids_path, std::filesystem::copy_options::overwrite_existing);
    return 0;
}

template DISKANN_DLLEXPORT void diskann_create_disk_layout<int8_t>(const std::string base_file,
                                                                   const std::string mem_index_file,
                                                                   const std::string output_file,
                                                                   const std::string reorder_data_file);
template DISKANN_DLLEXPORT void diskann_create_disk_layout<uint8_t>(const std::string base_file,
                                                                    const std::string mem_index_file,
                                                                    const std::string output_file,
                                                                    const std::string reorder_data_file);
template DISKANN_DLLEXPORT void diskann_create_disk_layout<float>(const std::string base_file, const std::string mem_index_file,
                                                                  const std::string output_file,
                                                                  const std::string reorder_data_file);

template DISKANN_DLLEXPORT void create_pageann_disk_layout<int8_t>(std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint64_t nnodes_per_sector, const float pq_cache_ratio, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<int8_t>> data_store, const size_t ndims, const uint32_t page_graph_degree, const std::string &pq_compressed_all_nodes_path, const std::string &output_file,
                                                           const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t fill_L, bool enable_spare_fill);
template DISKANN_DLLEXPORT void create_pageann_disk_layout<uint8_t>(std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint64_t nnodes_per_sector, const float pq_cache_ratio, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<uint8_t>> data_store, const size_t ndims, const uint32_t page_graph_degree, const std::string &pq_compressed_all_nodes_path, const std::string &output_file,
                                                            const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t fill_L, bool enable_spare_fill);
template DISKANN_DLLEXPORT void create_pageann_disk_layout<float>(std::unique_ptr<uint8_t[]>& reorder_pq_data_buff, tsl::robin_set<uint32_t>& cached_PQ_nodes, const uint64_t nnodes_per_sector, const float pq_cache_ratio, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<float>> data_store, const size_t ndims, const uint32_t page_graph_degree, const std::string &pq_compressed_all_nodes_path, const std::string &output_file,
                                                          const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t fill_L, bool enable_spare_fill);

template DISKANN_DLLEXPORT void create_laann_disk_layout<int8_t>(const uint64_t num_vectors_per_sector, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<int8_t>> data_store, const size_t ndims, const uint32_t optimal_vector_degree, const uint32_t L, const std::string &final_index_prefix_path,
                                                           const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t vamana_R);
template DISKANN_DLLEXPORT void create_laann_disk_layout<uint8_t>(const uint64_t num_vectors_per_sector, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<uint8_t>> data_store, const size_t ndims, const uint32_t optimal_vector_degree, const uint32_t L, const std::string &final_index_prefix_path,
                                                            const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t vamana_R);
template DISKANN_DLLEXPORT void create_laann_disk_layout<float>(const uint64_t num_vectors_per_sector, std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<float>> data_store, const size_t ndims, const uint32_t optimal_vector_degree, const uint32_t L, const std::string &final_index_prefix_path,
                                                          const std::vector<std::vector<uint32_t>>& mergedNodes, const std::vector<uint32_t>& nodeToMegaNodeMap, const uint32_t vamana_R);

                                                          
template DISKANN_DLLEXPORT int8_t *load_warmup<int8_t>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                       uint64_t warmup_dim, uint64_t warmup_aligned_dim);
template DISKANN_DLLEXPORT uint8_t *load_warmup<uint8_t>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                         uint64_t warmup_dim, uint64_t warmup_aligned_dim);
template DISKANN_DLLEXPORT float *load_warmup<float>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                     uint64_t warmup_dim, uint64_t warmup_aligned_dim);

#ifdef EXEC_ENV_OLS
template DISKANN_DLLEXPORT int8_t *load_warmup<int8_t>(MemoryMappedFiles &files, const std::string &cache_warmup_file,
                                                       uint64_t &warmup_num, uint64_t warmup_dim,
                                                       uint64_t warmup_aligned_dim);
template DISKANN_DLLEXPORT uint8_t *load_warmup<uint8_t>(MemoryMappedFiles &files, const std::string &cache_warmup_file,
                                                         uint64_t &warmup_num, uint64_t warmup_dim,
                                                         uint64_t warmup_aligned_dim);
template DISKANN_DLLEXPORT float *load_warmup<float>(MemoryMappedFiles &files, const std::string &cache_warmup_file,
                                                     uint64_t &warmup_num, uint64_t warmup_dim,
                                                     uint64_t warmup_aligned_dim);
#endif

template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t, uint32_t>(
    std::unique_ptr<diskann::PQFlashIndex<int8_t, uint32_t>> &pFlashIndex, int8_t *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);
template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t, uint32_t>(
    std::unique_ptr<diskann::PQFlashIndex<uint8_t, uint32_t>> &pFlashIndex, uint8_t *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);
template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float, uint32_t>(
    std::unique_ptr<diskann::PQFlashIndex<float, uint32_t>> &pFlashIndex, float *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);

template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t, uint16_t>(
    std::unique_ptr<diskann::PQFlashIndex<int8_t, uint16_t>> &pFlashIndex, int8_t *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);
template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t, uint16_t>(
    std::unique_ptr<diskann::PQFlashIndex<uint8_t, uint16_t>> &pFlashIndex, uint8_t *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);
template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float, uint16_t>(
    std::unique_ptr<diskann::PQFlashIndex<float, uint16_t>> &pFlashIndex, float *tuning_sample,
    uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads, uint32_t start_bw);

template DISKANN_DLLEXPORT int build_disk_index<int8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                                  const char *indexBuildParameters,
                                                                  diskann::Metric compareMetric, bool use_opq,
                                                                  const std::string &codebook_prefix, bool use_filters,
                                                                  const std::string &label_file,
                                                                  const std::string &universal_label,
                                                                  const uint32_t filter_threshold, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_disk_index<uint8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                                   const char *indexBuildParameters,
                                                                   diskann::Metric compareMetric, bool use_opq,
                                                                   const std::string &codebook_prefix, bool use_filters,
                                                                   const std::string &label_file,
                                                                   const std::string &universal_label,
                                                                   const uint32_t filter_threshold, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_disk_index<float, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                                 const char *indexBuildParameters,
                                                                 diskann::Metric compareMetric, bool use_opq,
                                                                 const std::string &codebook_prefix, bool use_filters,
                                                                 const std::string &label_file,
                                                                 const std::string &universal_label,
                                                                 const uint32_t filter_threshold, const uint32_t Lf);
// LabelT = uint16
template DISKANN_DLLEXPORT int build_disk_index<int8_t, uint16_t>(const char *dataFilePath, const char *indexFilePath,
                                                                  const char *indexBuildParameters,
                                                                  diskann::Metric compareMetric, bool use_opq,
                                                                  const std::string &codebook_prefix, bool use_filters,
                                                                  const std::string &label_file,
                                                                  const std::string &universal_label,
                                                                  const uint32_t filter_threshold, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_disk_index<uint8_t, uint16_t>(const char *dataFilePath, const char *indexFilePath,
                                                                   const char *indexBuildParameters,
                                                                   diskann::Metric compareMetric, bool use_opq,
                                                                   const std::string &codebook_prefix, bool use_filters,
                                                                   const std::string &label_file,
                                                                   const std::string &universal_label,
                                                                   const uint32_t filter_threshold, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_disk_index<float, uint16_t>(const char *dataFilePath, const char *indexFilePath,
                                                                 const char *indexBuildParameters,
                                                                 diskann::Metric compareMetric, bool use_opq,
                                                                 const std::string &codebook_prefix, bool use_filters,
                                                                 const std::string &label_file,
                                                                 const std::string &universal_label,
                                                                 const uint32_t filter_threshold, const uint32_t Lf);

template DISKANN_DLLEXPORT int build_page_graph<float>(const std::string &index_prefix_path, const std::string &data_file_to_use, const uint32_t min_degree_per_node,
                                                const uint32_t R, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, float memBudgetInGB, bool full_ooc, const std::string &new_to_old_ids_map_file, bool enable_spare_fill);

template DISKANN_DLLEXPORT int build_page_graph<uint8_t>(const std::string &index_prefix_path, const std::string &data_file_to_use, const uint32_t min_degree_per_node,
                                                    const uint32_t R, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, float memBudgetInGB, bool full_ooc, const std::string &new_to_old_ids_map_file, bool enable_spare_fill);

template DISKANN_DLLEXPORT int build_page_graph<int8_t>(const std::string &index_prefix_path, const std::string &data_file_to_use, const uint32_t min_degree_per_node,
                                                        const uint32_t R, const uint32_t num_pq_chunks_32, diskann::Metric compareMetric, float memBudgetInGB, bool full_ooc, const std::string &new_to_old_ids_map_file, bool enable_spare_fill);

                                                        
    // Template instantiations for build_laann_graph
template DISKANN_DLLEXPORT int build_laann_graph<float>(const std::string &index_prefix_path,
    const std::string &data_file_to_use, const uint32_t R, const uint32_t grouping_L, const uint32_t fill_L,
    const uint32_t expected_degree_per_vector, const uint32_t num_pq_chunks_32,
    diskann::Metric compareMetric, bool use_greedy_grouping, const std::string &new_to_old_ids_map_file);

template DISKANN_DLLEXPORT int build_laann_graph<uint8_t>(const std::string &index_prefix_path,
    const std::string &data_file_to_use, const uint32_t R, const uint32_t grouping_L, const uint32_t fill_L,
    const uint32_t expected_degree_per_vector, const uint32_t num_pq_chunks_32,
    diskann::Metric compareMetric, bool use_greedy_grouping, const std::string &new_to_old_ids_map_file);

template DISKANN_DLLEXPORT int build_laann_graph<int8_t>(const std::string &index_prefix_path,
    const std::string &data_file_to_use, const uint32_t R, const uint32_t grouping_L, const uint32_t fill_L,
    const uint32_t expected_degree_per_vector, const uint32_t num_pq_chunks_32,
    diskann::Metric compareMetric, bool use_greedy_grouping, const std::string &new_to_old_ids_map_file);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodes<float>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<float>> data_store,
    const uint32_t npts, const uint32_t ndim, const std::string index_prefix_path,
    const uint32_t R, const uint32_t num_pq_chunk, std::vector<std::vector<uint32_t>>& mergedNodes,
    std::vector<uint32_t>& nodeToMegaNodeMap, std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodes<uint8_t>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<uint8_t>> data_store,
    const uint32_t npts, const uint32_t ndim, const std::string index_prefix_path,
    const uint32_t R, const uint32_t num_pq_chunk, std::vector<std::vector<uint32_t>>& mergedNodes,
    std::vector<uint32_t>& nodeToMegaNodeMap, std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodes<int8_t>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<int8_t>> data_store,
    const uint32_t npts, const uint32_t ndim, const std::string index_prefix_path,
    const uint32_t R, const uint32_t num_pq_chunk, std::vector<std::vector<uint32_t>>& mergedNodes,
    std::vector<uint32_t>& nodeToMegaNodeMap, std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodesGreedy<float>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<float>> data_store,
    const uint32_t npts, const std::string index_prefix_path, const uint32_t L,
    std::vector<std::vector<uint32_t>>& mergedNodes, std::vector<uint32_t>& nodeToMegaNodeMap,
    std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodesGreedy<uint8_t>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<uint8_t>> data_store,
    const uint32_t npts, const std::string index_prefix_path, const uint32_t L,
    std::vector<std::vector<uint32_t>>& mergedNodes, std::vector<uint32_t>& nodeToMegaNodeMap,
    std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT size_t mergeNodesIntoMegaNodesGreedy<int8_t>(const uint64_t megaNode_capacity,
    std::shared_ptr<InMemOOCGraphStore> graph_store, std::shared_ptr<InMemOOCDataStore<int8_t>> data_store,
    const uint32_t npts, const std::string index_prefix_path, const uint32_t L,
    std::vector<std::vector<uint32_t>>& mergedNodes, std::vector<uint32_t>& nodeToMegaNodeMap,
    std::vector<uint32_t>& new_to_original_map);

template DISKANN_DLLEXPORT void optimal_vamana_graph_degree<float>(const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t num_pq_chunks_32, float memBudgetInGB, bool full_ooc);
template DISKANN_DLLEXPORT void optimal_vamana_graph_degree<uint8_t>(const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t num_pq_chunks_32, float memBudgetInGB, bool full_ooc);
template DISKANN_DLLEXPORT void optimal_vamana_graph_degree<int8_t>(const std::string &data_file_to_use, const uint32_t min_degree_per_node, const uint32_t num_pq_chunks_32, float memBudgetInGB, bool full_ooc);

template DISKANN_DLLEXPORT int build_merged_vamana_index<int8_t, uint32_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_merged_vamana_index<float, uint32_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_merged_vamana_index<uint8_t, uint32_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
// Label=16_t
template DISKANN_DLLEXPORT int build_merged_vamana_index<int8_t, uint16_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_merged_vamana_index<float, uint16_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
template DISKANN_DLLEXPORT int build_merged_vamana_index<uint8_t, uint16_t>(
    std::string base_file, diskann::Metric compareMetric, uint32_t L, uint32_t R, double sampling_rate,
    double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads, bool use_filters, const std::string &label_file,
    const std::string &labels_to_medoids_file, const std::string &universal_label, const uint32_t Lf);
}; // namespace diskann
