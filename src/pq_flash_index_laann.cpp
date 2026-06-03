// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index_laann.h"
#include "cosine_similarity.h"
#include "trace_logger.h"
#include <unordered_map>
#include <nmmintrin.h>
#include <filter_utils.h>
#include <numeric>
#include <fstream>

#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <pthread.h>
#include <sched.h>

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#ifdef USE_BING_INFRA
#warning "USE_BING_INFRA is enabled"
#endif

namespace diskann {
    std::unique_ptr<TraceLogger> g_trace_logger = nullptr;
}

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndexLAANN<T, LabelT>::PQFlashIndexLAANN(std::shared_ptr<AlignedFileReader> &dataReader, diskann::Metric m)
    : dataReader(dataReader), metric(m), _thread_data(nullptr)
{
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndexLAANN<T, LabelT>::~PQFlashIndexLAANN()
{
    // Navigation graph data cleanup
    if (_nav_nbrs != nullptr) {
        delete[] _nav_nbrs;
        _nav_nbrs = nullptr;
    }

    // Page cache data cleanup
    if (_cached_page_data != nullptr) {
        diskann::aligned_free(_cached_page_data);
        _cached_page_data = nullptr;
    }

    if (_cached_pq_buff != nullptr){
        delete[] _cached_pq_buff;
    }
    
    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->dataReader->deregister_all_threads();
        dataReader->close();
        diskann::cout << "Cleared scratch" << std::endl;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
    
    diskann::cout << "Destroyed PQ Flash Index Object" << std::endl;
}

template <typename T, typename LabelT> inline T *PQFlashIndexLAANN<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t estimated_visited_vectors, uint64_t beam_width)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, estimated_visited_vectors, _megaNode_capacity, false, beam_width);//each data includes a SSDQueryScratch and one IOContext

#ifdef __linux__
            try {
                uint32_t queue_depth = 128;
                data->scratch.async_reader = new LinuxAsyncFileReader(_disk_index_file, queue_depth);
                if (!data->scratch.async_reader->init()) {
                    delete data->scratch.async_reader;
                    data->scratch.async_reader = nullptr;
                    diskann::cerr << "ERROR: Failed to initialize async I/O for thread" << std::endl;
                }
                _use_async_io = true;
            } catch (const std::exception &e) {
                diskann::cerr << "ERROR: Async reader creation failed: " << e.what() << std::endl;
                data->scratch.async_reader = nullptr;
            }
#endif         
                  
            //this->dataReader->register_thread();
            //NOTE: no need to change the AlignedReader file cuz ctx_map belongs to each reader object
            //data->data_ctx = this->dataReader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}


template <typename T, typename LabelT>
std::vector<bool> PQFlashIndexLAANN<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<uint32_t *> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _num_vectors_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
///MARK: need to divide by 2
    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto page_id = node_ids[i];
        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        //first page is for the meta data so we increment it by 1
        ///MARK: this offset is right. the first page is used to store metadata
        read.offset = (page_id + 1) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    //get one SSDThreadData object from _thread_data
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();//the object, SSDThreadData , we get
    IOContext &data_ctx = this_thread_data->data_ctx;
    dataReader->read(read_reqs, data_ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*data_ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        //original code: //char *node_buf = offset_to_node((char *)read_reqs[i].buf, page_ids[i]);
        char *node_buf = (char *)read_reqs[i].buf; //because the representative node of the page is stored as the first node in the very begining

        if (coord_buffers[i] != nullptr)
        {
            //this cast char * to T*
            T *node_coords = offset_to_node_coords(node_buf);
            //_bytes_per_vector was calculated as dim * sizof(T)
            //the third parameter: the number of bytes to copy from the source to the destination.
            memcpy(coord_buffers[i], node_coords, _num_vectors_per_sector * _bytes_per_vector);
        }

        ///NOTE: this only used for cache_bfs_level -- we need to get the ids of nbr pages
        // if (nbr_buffers[i] != nullptr)
        // {
        //     uint64_t nnbrs = 64;
        //     uint64_t each_nbr_space = (uint64_t)(sizeof(uint32_t) + _n_chunks * sizeof(uint8_t)); //_n_chunks should be 12
        //     for(uint8_t k = 0; k < nnbrs; k++){
        //         uint32_t *node_nhood = (uint32_t *)(node_buf + _bytes_per_vector * _num_vectors_per_sector + k * each_nbr_space);
        //         memcpy(nbr_buffers[i] + k, node_nhood, sizeof(uint32_t));
        //     }
        // }
    }

    aligned_free(buf);

    return retval;
}


template <typename T, typename LabelT>
std::vector<bool> PQFlashIndexLAANN<T, LabelT>::read_mega_node(const std::vector<uint32_t> &mega_nodes_ids, std::vector<T *> &coord_buffers, std::vector<uint32_t *> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    read_reqs.reserve(mega_nodes_ids.size());
    std::vector<bool> retval(mega_nodes_ids.size(), true);
    char *buf = nullptr;
    alloc_aligned((void **)&buf, mega_nodes_ids.size() * _num_sectors_per_node * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
    std::memset(buf, 0, mega_nodes_ids.size() * defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < mega_nodes_ids.size(); ++i)
    {
        AlignedRead read;
        read.len = defaults::SECTOR_LEN;
        read.buf = buf + i * defaults::SECTOR_LEN;
        read.offset = (1 + mega_nodes_ids[i]) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &data_ctx = this_thread_data->data_ctx;
    dataReader->read(read_reqs, data_ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*data_ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif
        // Validate destination buffer
        if (coord_buffers[i] == nullptr) {
            diskann::cerr << "Error: coord_buffers[" << i << "] is null" << std::endl;
            retval[i] = false;
            continue;
        }

        T *node_coords = offset_to_node_coords((char *)read_reqs[i].buf);//this just cast char * to T*
        memcpy(coord_buffers[i], node_coords, _all_vectors_space_per_pageNode);

        if (nbr_buffers[i] == nullptr) {
            diskann::cerr << "Error: nbr_buffers[" << i << "] is null" << std::endl;
            retval[i] = false;
            continue;
        }

        uint32_t* node_nbrs = reinterpret_cast<uint32_t*>((char *)read_reqs[i].buf + _all_vectors_space_per_pageNode);
        std::memcpy(nbr_buffers[i], node_nbrs, _all_nbrs_space_per_pageNode);
    }

    aligned_free(buf);
    return retval;
}

template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::load_sequential_page_cache(uint32_t num_pages_to_cache)
{
    if (num_pages_to_cache == 0) {
        diskann::cout << "No pages to cache." << std::endl;
        return;
    }

    // Clamp to max available pages
    if (num_pages_to_cache > _num_mega_nodes) {
        num_pages_to_cache = static_cast<uint32_t>(_num_mega_nodes);
    }

    diskann::cout << "Loading " << num_pages_to_cache << " sequential pages into cache..." << std::flush;

    // Allocate aligned memory for page cache
    size_t cache_size = static_cast<size_t>(num_pages_to_cache) * defaults::SECTOR_LEN;
    diskann::alloc_aligned((void**)&_cached_page_data, cache_size, defaults::SECTOR_LEN);

    // Open disk index file for reading
    std::ifstream index_reader(_disk_index_file, std::ios::binary);
    if (!index_reader) {
        diskann::cerr << "Failed to open index file: " << _disk_index_file << std::endl;
        diskann::aligned_free(_cached_page_data);
        _cached_page_data = nullptr;
        return;
    }

    // Skip metadata sector (first sector), read data pages starting from sector 1
    index_reader.seekg(defaults::SECTOR_LEN);

    // Read all pages in one sequential read for maximum efficiency
    index_reader.read(_cached_page_data, cache_size);

    if (!index_reader) {
        diskann::cerr << "Failed to read pages from disk." << std::endl;
        diskann::aligned_free(_cached_page_data);
        _cached_page_data = nullptr;
        index_reader.close();
        return;
    }

    index_reader.close();
    _num_cached_pages = num_pages_to_cache;

    diskann::cout << "Done. Cached " << _num_cached_pages << " pages ("
                  << (cache_size / (1024 * 1024)) << " MB)." << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::load_order_file_page_cache(const std::string &order_file, uint32_t num_pages_to_cache)
{
    if (num_pages_to_cache == 0) {
        diskann::cout << "No pages to cache." << std::endl;
        return;
    }

    // Read order file: [num_pages uint32][page_id uint32, freq uint32]...
    std::ifstream ofs(order_file, std::ios::binary);
    if (!ofs.is_open()) {
        diskann::cerr << "Failed to open order file: " << order_file << std::endl;
        return;
    }

    uint32_t total_pages_in_file;
    ofs.read(reinterpret_cast<char*>(&total_pages_in_file), sizeof(uint32_t));

    num_pages_to_cache = std::min(num_pages_to_cache, total_pages_in_file);
    num_pages_to_cache = std::min(num_pages_to_cache, static_cast<uint32_t>(_num_mega_nodes));

    diskann::cout << "Loading " << num_pages_to_cache << " pages by order file into cache..." << std::flush;

    // Read top num_pages_to_cache page IDs (skip freq field)
    std::vector<uint32_t> page_ids(num_pages_to_cache);
    for (uint32_t i = 0; i < num_pages_to_cache; i++) {
        uint32_t page_id, freq;
        ofs.read(reinterpret_cast<char*>(&page_id), sizeof(uint32_t));
        ofs.read(reinterpret_cast<char*>(&freq),    sizeof(uint32_t));
        page_ids[i] = page_id;
    }
    ofs.close();

    // Allocate aligned cache buffer (one slot per page)
    size_t cache_size = static_cast<size_t>(num_pages_to_cache) * defaults::SECTOR_LEN;
    diskann::alloc_aligned((void**)&_cached_page_data, cache_size, defaults::SECTOR_LEN);

    // Open disk index for random reads
    std::ifstream index_reader(_disk_index_file, std::ios::binary);
    if (!index_reader) {
        diskann::cerr << "Failed to open index file: " << _disk_index_file << std::endl;
        diskann::aligned_free(_cached_page_data);
        _cached_page_data = nullptr;
        return;
    }

    // For each page_id: seek to its position in the index (non-contiguous), read into slot i
    for (uint32_t i = 0; i < num_pages_to_cache; i++) {
        uint32_t page_id = page_ids[i];
        uint64_t disk_offset = static_cast<uint64_t>(1 + page_id) * defaults::SECTOR_LEN;
        index_reader.seekg(static_cast<std::streamoff>(disk_offset));
        index_reader.read(_cached_page_data + static_cast<size_t>(i) * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        if (!index_reader) {
            diskann::cerr << "Failed to read page " << page_id << " from disk." << std::endl;
            diskann::aligned_free(_cached_page_data);
            _cached_page_data = nullptr;
            return;
        }
        _cached_page_id_to_slot[page_id] = i;
    }

    index_reader.close();
    _num_cached_pages = num_pages_to_cache;
    _use_order_file_cache = true;

    diskann::cout << "Done. Cached " << _num_cached_pages << " pages ("
                  << (cache_size / (1024 * 1024)) << " MB) via order file." << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::profile_page_frequency(std::string sample_bin, uint64_t l_search,
                                                              uint64_t beam_width, uint32_t nthreads,
                                                              const std::string &output_file, const uint32_t nav_L)
{
    diskann::cout << "=== PAGE FREQUENCY PROFILING ===" << std::endl;

    // Initialize visit counters
    this->_count_visited_megaNodes = true;
    this->_mega_node_visit_counter.clear();
    this->_mega_node_visit_counter.resize(this->_num_mega_nodes);
    for (uint32_t i = 0; i < _mega_node_visit_counter.size(); i++)
    {
        this->_mega_node_visit_counter[i].first = i;
        this->_mega_node_visit_counter[i].second = 0;
    }

    // Load sample queries
    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
        diskann::cout << "Loaded " << sample_num << " sample queries from: " << sample_bin << std::endl;
    }
    else
    {
        diskann::cerr << "Error: Sample bin file not found: " << sample_bin << std::endl;
        return;
    }

    // Run searches to collect page visit statistics
    std::vector<float> query_result_dists(1 * sample_num);
    std::vector<uint64_t> query_result_ids(1 * sample_num);
    auto stats = new diskann::QueryStats[sample_num];

    diskann::cout << "Running " << sample_num << " sample queries for profiling..." << std::endl;
//query, k, l, ids, dists, cache B, normal B, use cache priotity, use pipeline, secondary cache B, nav_L, use reorder data, states
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        laann_search(samples + (i * sample_aligned_dim), 1, l_search,
                        query_result_ids.data() + i * 1, query_result_dists.data() + i * 1,
                        beam_width, false, true, 0,
                        // nav_L, use_reorder_data=false, retset_capacity_ratio=1.0f,
                        // stable_rank_threshold=l_search (large value → convergence never triggers during profiling),
                        // beamwidth_spike_ratio=0.5f, beam_decay_ratio=0.9f, stats
                        nav_L, false, 1.0f, static_cast<uint32_t>(l_search), 0.5f, 0.9f, stats + i);
    }

    delete[] stats;
    diskann::aligned_free(samples);

    // Disable counting
    this->_count_visited_megaNodes = false;

    // Sort by frequency (descending)
    diskann::cout << "Sorting pages by visit frequency..." << std::endl;
    std::sort(this->_mega_node_visit_counter.begin(), _mega_node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });

    // Write to binary file
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs.is_open())
    {
        diskann::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return;
    }

    // Write header: num_pages (uint32_t)
    uint32_t num_pages = static_cast<uint32_t>(this->_mega_node_visit_counter.size());
    ofs.write(reinterpret_cast<const char*>(&num_pages), sizeof(uint32_t));

    // Write sorted (page_id, frequency) pairs
    for (const auto& pf : this->_mega_node_visit_counter)
    {
        ofs.write(reinterpret_cast<const char*>(&pf.first), sizeof(uint32_t));   // page_id
        ofs.write(reinterpret_cast<const char*>(&pf.second), sizeof(uint32_t));  // frequency
    }
    ofs.close();

    diskann::cout << "Page frequency profile written to: " << output_file << std::endl;
    diskann::cout << "Total pages: " << num_pages << std::endl;

    // Print statistics
    uint64_t total_visits = 0;
    uint32_t pages_with_visits = 0;
    for (const auto& pf : this->_mega_node_visit_counter)
    {
        total_visits += pf.second;
        if (pf.second > 0) pages_with_visits++;
    }

    diskann::cout << "Total page visits: " << total_visits << std::endl;
    diskann::cout << "Pages with visits: " << pages_with_visits << " / " << num_pages
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * pages_with_visits / num_pages) << "%)" << std::endl;

    // Print top 10 most visited pages
    diskann::cout << "\nTop 10 most visited pages:" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, _mega_node_visit_counter.size()); i++)
    {
        diskann::cout << "  Page " << std::setw(8) << _mega_node_visit_counter[i].first
                      << ": " << std::setw(8) << _mega_node_visit_counter[i].second << " visits" << std::endl;
    }

    diskann::cout << "=== PROFILING COMPLETE ===" << std::endl;
}

template <typename T, typename LabelT> int PQFlashIndexLAANN<T, LabelT>::load(uint32_t num_threads, const char *index_prefix, const std::string &pq_path_prefix, uint32_t beam_width)
{

    ///NOTE: this file names are matched with those hardcoded file names in disk_utils.cpp, build_disk_index function.
    std::string pq_table_bin = (pq_path_prefix.empty() ? std::string(index_prefix) : pq_path_prefix) + "_pq_pivots.bin";

    ///MARK: we dont need to load pq_compressed_vectors here
    //std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string pq_compressed_reorder_file = (pq_path_prefix.empty() ? std::string(index_prefix) : pq_path_prefix) + "_reorder_pq_compressed.bin";
    return load_from_separate_paths(num_threads, index_prefix, pq_table_bin.c_str(),
                                    pq_compressed_reorder_file.c_str(), beam_width);
}

template <typename T, typename LabelT>
int PQFlashIndexLAANN<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *pq_compressed_reorder_file_path, uint32_t beam_width)
{
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_reorder_file = pq_compressed_reorder_file_path;
    std::string _disk_index_file = std::string(index_filepath) + ".index";

    size_t pq_file_dim, pq_file_num_centroids;

    ///NOTE: this is right since it uses offset of METADATA_SIZE --- at which the full pq pivots are written
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);//definition and implementation is in utils.h

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    size_t nchunks_u64;//get it from pq_table_bin 

    ///MARK: load data into _pq_table's fields including table, table_tr, centroid, n_chunks, chunk_offsets
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    this->_n_chunks = nchunks_u64; 

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

// read index metadata

    std::ifstream index_metadata(_disk_index_file, std::ios::binary);

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);//this is the number of meta info and the dimension of each data info
    READ_U32(index_metadata, nc);

    if (!index_metadata) {
        std::cerr << "Error opening file: " << _disk_index_file << std::endl;
        return -1;
    }

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
//use other traditional ways to read this data
    READ_U64(index_metadata, disk_nnodes); // 0: npts (actual, not aligned)
    READ_U64(index_metadata, disk_ndims); // 1: ndims
    
    // if (disk_nnodes != _num_points)
    // {
    //     diskann::cout << "Mismatch in #points for compressed data file and disk "
    //                      "index file: "
    //                   << disk_nnodes << " vs " << _num_points << std::endl;
    //     return -1;
    // }

    this->_data_dim = disk_ndims;
    this->_bytes_per_vector = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(disk_ndims, 8);//to align with 8 is cuz AXV for computing distance will load 8 float at a time
    this->_num_points = disk_nnodes;
    
    //this->_n_chunks = 12;//
    this->_n_chunks = nchunks_u64;
    diskann::cout << "#points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

/// LAANN metadata format: [0]=npts, [1]=ndims, [2]=medoid, [3]=each_node_space,
///   [4]=megaNode_capacity, [5]=num_pages, [6]=optimal_vector_degree, [7]=vamana_R
    size_t medoid_id_on_file;
    uint64_t each_node_space, num_pages_meta, vamana_R_meta;
    READ_U64(index_metadata, medoid_id_on_file);           // 2: medoid
    READ_U64(index_metadata, each_node_space);             // 3: each_node_space = ndims * sizeof(T)
    READ_U64(index_metadata, _megaNode_capacity);          // 4: megaNode_capacity
    READ_U64(index_metadata, num_pages_meta);              // 5: num_pages (skip)
    READ_U64(index_metadata, _max_page_degree);            // 6: mega_node_degree (real max neighbors per page)
    READ_U64(index_metadata, vamana_R_meta);               // 7: vamana_R (skip)
    _max_node_len = each_node_space;                       // vector stride; drives _num_sectors_per_node

    _num_vectors_per_sector = _megaNode_capacity;  // For PageANN, same thing
    _num_mega_nodes = (_num_points + _megaNode_capacity - 1) / _megaNode_capacity;  // Calculate from num_points
    _each_pq_space = (uint64_t)(_n_chunks * sizeof(uint8_t));
    // PageANN: all vectors including centroid are stored
    _pageNode_vectors_size_in_T = _megaNode_capacity * _data_dim;//coords_stride
    _all_vectors_space_per_pageNode = _pageNode_vectors_size_in_T * sizeof(T);
    _pageNode_nbrs_size_in_u32 = _max_page_degree;
    _all_nbrs_space_per_pageNode = _pageNode_nbrs_size_in_u32 * sizeof(uint32_t);
    _num_sectors_per_node = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    _last_mega_node_index = _num_mega_nodes - 1;

    // Precompute number of vectors in last page (PageANN: all vectors including centroid)
    uint64_t last_page_count = _num_points % _megaNode_capacity;
    if (last_page_count == 0) {
        last_page_count = _megaNode_capacity;
    }
    _last_page_vector_count = last_page_count;

    diskann::cout << "Disk-Index File Meta-data (PageANN format): " << std::endl;
    diskann::cout << "medoid's page ID: " << medoid_id_on_file << std::endl;
    diskann::cout << "num of pages: " << this->_num_mega_nodes << std::endl;
    diskann::cout << "PQ dim in use: " << _n_chunks << std::endl;
    diskann::cout << "# vectors per sector: " << _num_vectors_per_sector << std::endl;
    diskann::cout << "_all_vectors_space_per_pageNode: " << _all_vectors_space_per_pageNode << std::endl;
    diskann::cout << "SECTOR_LEN: " << defaults::SECTOR_LEN << std::endl;
    diskann::cout << "mega node capacity: " << _megaNode_capacity << std::endl;
    diskann::cout << "_num_sectors_per_node: " << _num_sectors_per_node << std::endl;
    diskann::cout << "max node len (bytes): " << _max_node_len << std::endl;
    diskann::cout << "actual max page-node graph degree: " << _max_page_degree << std::endl;

    //MAX_GRAPH_DEGREE is 512
    if (_max_page_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    
    index_metadata.close();

    //begin to load cached PQ data
    //std::string pq_compressed_reorder_file = std::string(index_filepath) + "_reorder_pq_compressed.bin";
    std::ifstream inPQFile(pq_compressed_reorder_file, std::ios::binary);
    if (!inPQFile.is_open()) {
        throw std::runtime_error("Failed to open reorder compressed PQ data file for reading.");
    }

    // LAANN PQ file format: [points_num (uint32_t)][num_pq_chunks (uint32_t)][PQ data...]
    uint32_t points_num, num_pq_chunks;
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("VmRSS", 0) == 0 || line.rfind("VmSize", 0) == 0) {
                std::cout << "[" << "before loading PQ" << "] " << line << std::endl;
            }
        }
    }

    inPQFile.read(reinterpret_cast<char*>(&points_num), sizeof(points_num));
    inPQFile.read(reinterpret_cast<char*>(&num_pq_chunks), sizeof(num_pq_chunks));

    std::cout << "points_num: " << points_num << std::endl;
    std::cout << "num_pq_chunks: " << num_pq_chunks << std::endl;
    diskann::cout << "Loading PQ data to memory..." << std::endl;

    uint64_t buffer_size = static_cast<uint64_t>(points_num) * num_pq_chunks;
    _cached_pq_buff = new uint8_t[buffer_size];
    inPQFile.read(reinterpret_cast<char*>(_cached_pq_buff), buffer_size * sizeof(uint8_t));
    diskann::cout << "PQ buffer size: " << buffer_size << "B" << std::endl;
    inPQFile.close();

    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("VmRSS", 0) == 0 || line.rfind("VmSize", 0) == 0) {
                std::cout << "[" << "after loading PQ values" << "] " << line << std::endl;
            }
        }
    }

    // open AlignedFileReader handle to index_file
    std::string index_fname(_disk_index_file);
    dataReader->open(index_fname);

    this->setup_thread_data(num_threads, defaults::ESTIMATED_VISITED_VECTORS, beam_width);
    this->_max_nthreads = num_threads;

    _num_medoids = 1;
    _medoids = new uint32_t[1];
    _medoids[0] = (uint32_t)(medoid_id_on_file);
    
    diskann::cout << "done of loading......" << std::endl;
    return 0;
}

// Helper function to generate sample indices within a page using priority order:
// top (0), last (capacity-1), second-to-last (capacity-2), ...
static std::vector<uint32_t> generate_sample_indices(uint32_t page_capacity, uint32_t samples_per_page) {
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

//
template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::load_nav_neighbors(const char *index_prefix)
{
    // LAANN format: *_nav_graph.index
    std::string nav_file = std::string(index_prefix) + "_nav_graph.index";

    // Check if file exists
    std::ifstream nav_test(nav_file);
    if (!nav_test.good()) {
        diskann::cerr << "Navigation graph file not found: " << nav_file << std::endl;
        _nav_graph_loaded = false;
        return;
    }
    nav_test.close();

    diskann::cout << "Loading navigation graph from: " << nav_file << std::endl;

    try {
        std::ifstream graph_in(nav_file, std::ios::binary);

        // Read metadata: [num_sampled, nav_degree, entry_point, samples_per_page, meganode_capacity, num_total_pages]
        // Field 5 (num_total_pages) is new — 0 means normal mode (all pages sampled).
        // Older nav graphs without field 5 are treated as normal mode.
        uint64_t nav_metadata[6] = {0};
        graph_in.read((char*)nav_metadata, 5 * sizeof(uint64_t));
        graph_in.read((char*)&nav_metadata[5], sizeof(uint64_t));  // 0 if not present (eof)

        uint64_t num_sampled              = nav_metadata[0];
        uint64_t nav_degree               = nav_metadata[1];
        uint64_t entry_point_original_id  = nav_metadata[2];
        uint64_t samples_per_page         = nav_metadata[3];
        uint64_t meganode_capacity        = nav_metadata[4];
        uint64_t num_total_pages          = nav_metadata[5];  // 0 = normal, >0 = page subsampling

        _nav_num_points          = static_cast<uint32_t>(num_sampled);
        _nav_uniform_degree      = static_cast<uint32_t>(nav_degree);
        _nav_medoid_original_id  = static_cast<uint32_t>(entry_point_original_id);
        _nav_samples_per_page    = static_cast<uint32_t>(samples_per_page);
        _nav_total_pages         = static_cast<uint32_t>(num_total_pages);

        if (meganode_capacity != _megaNode_capacity) {
            diskann::cerr << "Warning: Nav graph meganode_capacity (" << meganode_capacity
                         << ") != index meganode_capacity (" << _megaNode_capacity << ")" << std::endl;
        }

        bool page_subsampling = (_nav_total_pages > 0 && _nav_num_points < _nav_total_pages);

        diskann::cout << "  Navigation graph metadata:" << std::endl;
        diskann::cout << "    Num sampled vectors: " << _nav_num_points << std::endl;
        diskann::cout << "    Uniform degree: " << _nav_uniform_degree << std::endl;
        diskann::cout << "    Entry point (original ID): " << _nav_medoid_original_id << std::endl;
        diskann::cout << "    MegaNode capacity: " << meganode_capacity << std::endl;

        if (page_subsampling) {
            // Reconstruct selected page IDs from the same formula used during build
            _nav_selected_pages.clear();
            _nav_selected_pages.reserve(_nav_num_points);
            uint32_t k = _nav_num_points, N = _nav_total_pages;
            if (k == 1) {
                _nav_selected_pages.push_back(0);
            } else {
                for (uint32_t i = 0; i < k; ++i)
                    _nav_selected_pages.push_back((uint64_t)i * (N - 1) / (k - 1));
            }
            diskann::cout << "    Mode: page subsampling (" << k << " pages out of " << N << ")" << std::endl;
        } else {
            _nav_selected_pages.clear();
            _nav_most_page_sample_indices = generate_sample_indices(static_cast<uint32_t>(_megaNode_capacity), _nav_samples_per_page);
            uint32_t vectors_in_last_page = static_cast<uint32_t>(_num_points % _megaNode_capacity);
            if (vectors_in_last_page == 0) vectors_in_last_page = static_cast<uint32_t>(_megaNode_capacity);
            uint32_t samples_in_last_page = std::min(_nav_samples_per_page, vectors_in_last_page);
            _nav_last_page_sample_indices = generate_sample_indices(vectors_in_last_page, samples_in_last_page);
            diskann::cout << "    Mode: full page sampling (" << _nav_samples_per_page << " per page)" << std::endl;
        }

        // Allocate flat buffer for navigation neighbors (all nodes have uniform degree)
        size_t total_nav_size = static_cast<size_t>(_nav_num_points) * _nav_uniform_degree;
        _nav_nbrs = new uint32_t[total_nav_size];

        // Read all neighbor lists in one read (fixed size per node)
        graph_in.read((char*)_nav_nbrs, total_nav_size * sizeof(uint32_t));

        graph_in.close();

        diskann::cout << "Navigation graph loaded successfully: "
                     << _nav_num_points << " nodes with uniform degree " << _nav_uniform_degree << std::endl;

        _nav_graph_loaded = true;

    } catch (const std::exception& e) {
        diskann::cerr << "Error loading navigation graph: " << e.what() << std::endl;
        _nav_graph_loaded = false;

        // Cleanup on error
        if (_nav_nbrs != nullptr) {
            delete[] _nav_nbrs;
            _nav_nbrs = nullptr;
        }
        _nav_most_page_sample_indices.clear();
        _nav_last_page_sample_indices.clear();
    }
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndexLAANN<T, LabelT>::get_num_points()
{
    return _num_points;
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndexLAANN<T, LabelT>::get_num_mega_nodes()
{
    return _num_mega_nodes;
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndexLAANN<T, LabelT>::get_num_cached_mega_nodes()
{
    return _num_cached_pages;
}

template <typename T, typename LabelT>
void PQFlashIndexLAANN<T, LabelT>::laann_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                                       uint64_t *res_ids, float *res_dists,
                                                       const uint64_t search_beam_width,
                                                       const bool use_look_ahead_search, const bool use_pipeline,
                                                       const uint32_t persistence_window_width,
                                                       const uint32_t nav_L, const bool use_reorder_data,
                                                       const float retset_capacity_ratio,
                                                       const uint32_t stable_rank_threshold,
                                                       const float beamwidth_spike_ratio, const float beam_decay_ratio, QueryStats *stats)
{
    const uint32_t io_limit = std::numeric_limits<uint32_t>::max();
    const uint64_t num_sectors_per_node = _num_vectors_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (search_beam_width > num_sectors_per_node * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__, __LINE__);
    const uint64_t max_IOs = search_beam_width;

    // Get scratch space from thread pool
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();
    query_scratch->reset();

    // Start timing for the entire query
    auto query_start = std::chrono::high_resolution_clock::now();

    // Detailed timing breakdown
    float tail_io_us = 0.0f;           // Blocking IO wait time (tail of IO flight, CPU idle)
    float io_flight_us = 0.0f;          // Total IO in-flight time (submit → completion, includes CPU overlap)
    float io_submit_us = 0.0f;          // I/O submission time
    float cached_proc_us = 0.0f;        // Time processing cached vectors
    float uncached_proc_us = 0.0f;      // Time processing uncached (disk) vectors
    float init_us = 0.0f;               // Initialization time
    float nav_us = 0.0f;                // Navigation phase time
    uint32_t n_cache_hits = 0;          // Number of cache hits

    // Aligned memory for query (required for SIMD operations)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // Normalize query for COSINE and MIPS metrics
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query[i];
            query_norm += query[i] * query[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query[i];
        }
    }
    pq_query_scratch->initialize(this->_data_dim, aligned_query_T);

    // Buffers for node data and distance calculations
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);
    
    // Precompute PQ distances
    _pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    tsl::robin_set<uint32_t> &visitedPages = query_scratch->expandedPages;   // main search
    tsl::robin_set<uint32_t> &seededPages  = query_scratch->seededPages;     // Phase 1.5 only
    NeighborPriorityQueue &retset = query_scratch->retset;//this is based pq distance
    // Enlarge retset when pipeline+cache is active: extra entries serve as IO-wait work reservoir (Step 3)
    const size_t retset_cap = (use_pipeline && _num_cached_pages > 0)
        ? static_cast<size_t>(l_search * retset_capacity_ratio)
        : l_search;
    retset.reserve(retset_cap);
    NeighborPriorityQueue &full_ret_queue = query_scratch->full_ret_queue;//this is based full-precision distance
    full_ret_queue.reserve(k_search);
    uint32_t full_precision_cmps = 0;
    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();

    auto &async_reader = query_scratch->async_reader;
    auto &io_buffer_pool = query_scratch->io_buffer_pool;//already reserved space in SSDQueryScratch
    auto &completed_io_map = query_scratch->completed_io_map;//already reserved space in SSDQueryScratch
    auto &pending_io_map = query_scratch->pending_io_map;//already reserved space in SSDQueryScratch

    uint32_t hops = 0;
    uint32_t num_ios = 0;
    uint32_t non_hub_requested = 0;  // Count non-hub vectors encountered

    // Pre-allocated buffer for neighbor filtering (reused across lambda calls)
    std::vector<uint32_t> candidate_nbrs_buf;
    candidate_nbrs_buf.reserve(defaults::MAX_GRAPH_DEGREE);

    // Reusable buffers for each iteration (reserve 2*max_IOs)
    std::vector<uint32_t> pages_to_read;
    pages_to_read.reserve(2 * max_IOs);
    std::vector<uint32_t> cached_pages;
    cached_pages.reserve(2 * max_IOs);

    // Delayed full-precision distance computation buffers. Strategy: When disk I/O completes, we immediately process graph neighbors (for search frontier expansion) but defer the expensive full-precision distance computation.
    // These deferred computations are then performed during the NEXT iteration's I/O wait, overlapping CPU work with disk latency for better throughput.
    // - deferred_proc_vec_uncached_page_ids: page IDs whose vector distances are pending (from uncached/disk pages)
    // - deferred_proc_uncached_vec_buffer: vector data copied from I/O buffers before release
    auto &deferred_proc_vec_uncached_page_ids = query_scratch->deferred_proc_vec_uncached_page_ids;
    deferred_proc_vec_uncached_page_ids.reserve(l_search * max_IOs);
    auto &deferred_proc_uncached_vec_buffer = query_scratch->deferred_proc_uncached_vec_buffer;
    deferred_proc_uncached_vec_buffer.resize(_all_vectors_space_per_pageNode * max_IOs * l_search);

    // Deferred cached page full-precision processing (similar to dynamic_page_search)
    // When only cached pages are selected (no I/O), defer full vector processing to next iteration's I/O wait
    std::vector<uint32_t> deferred_proc_vec_cached_page_ids;
    deferred_proc_vec_cached_page_ids.reserve(5 * max_IOs);

    // Lambda: Process megaNode (compute distances and expand neighbors)
    // Modified for PageANN disk graph layout
    auto process_pageNode = [&](uint32_t pageID, T* page_vectors_buf, char* nhood_buf) {

        if(page_vectors_buf != nullptr){
            // PageANN: all vectors including centroid at top
            const uint32_t first_node_id = static_cast<uint32_t>(pageID * _megaNode_capacity);
            uint64_t num_nodes = (pageID == _last_mega_node_index) ? _last_page_vector_count : _megaNode_capacity;
            // Full precision distance pass
            for (uint32_t k = 0; k < num_nodes; k++) {
                uint32_t node_id = first_node_id + k;
                T *cur_node_coords = page_vectors_buf + k * _data_dim;
                memcpy(data_buf, cur_node_coords, _bytes_per_vector);
                float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                full_ret_queue.insert(Neighbor(node_id, dist));
            }
            full_precision_cmps += static_cast<uint32_t>(num_nodes);
            if (stats != nullptr) stats->n_mega_nodes_processed += 1;
        }

        if(nhood_buf != nullptr){
            // LAANN layout: [total_num_nbrs (uint32_t)] [nbr_ids...]
            uint32_t total_num_nbrs = *reinterpret_cast<uint32_t*>(nhood_buf);
            uint32_t *neighbor_ids = reinterpret_cast<uint32_t*>(nhood_buf + sizeof(uint32_t));

            // Filter neighbors: skip those from already visited pages (same as PageANN)
            candidate_nbrs_buf.clear();
            for(size_t i = 0; i < total_num_nbrs; i++){
                uint32_t nbrID = neighbor_ids[i];
                uint32_t nbr_pageID = nbrID / _megaNode_capacity;
                if (visitedPages.find(nbr_pageID) == visitedPages.end()){
                    candidate_nbrs_buf.push_back(nbrID);
                }
            }

            // Compute PQ distances only for unvisited candidates
            for(size_t i = 0; i < candidate_nbrs_buf.size(); i++){
                //diskann::cout << candidate_nbrs_buf[i] << std::endl;
                memcpy(pq_coord_scratch + i * _each_pq_space, _cached_pq_buff + candidate_nbrs_buf[i] * _n_chunks, _each_pq_space);
            }

            diskann::pq_dist_lookup(pq_coord_scratch, candidate_nbrs_buf.size(), _n_chunks, pq_dists, dist_scratch);
            for (size_t i = 0; i < candidate_nbrs_buf.size(); i++){
                retset.insert(Neighbor(candidate_nbrs_buf[i], dist_scratch[i]));
            }
        }
    };

    // Lambda: Allocate aligned I/O buffer
    auto allocate_io_buffer = [&](size_t size) -> char* {
        for (auto &buf : io_buffer_pool) {
            if (!buf.in_use && buf.size >= size) {
                buf.in_use = true;
                return buf.data;
            }
        }
        char *new_buf = nullptr;
        int ret = posix_memalign((void**)&new_buf, 4096, size);
        if (ret != 0) {
            throw std::runtime_error("Failed to allocate aligned buffer: " + std::string(strerror(ret)));
        }
        io_buffer_pool.push_back({new_buf, size, true});
        return new_buf;
    };

    // Lambda: Release I/O buffer
    auto release_io_buffer = [&](char *buf) {
        for (auto &b : io_buffer_pool) {
            if (b.data == buf) {
                b.in_use = false;
                return;
            }
        }
    };

    // Lambda: Submit async read for uncached megaNode
    auto submit_async_read = [&](uint32_t pageID) -> bool {
        if (!_use_async_io || async_reader == nullptr) return false;

        uint64_t disk_offset = (1 + pageID) * defaults::SECTOR_LEN;
        size_t read_size = defaults::SECTOR_LEN;

        char *buf = allocate_io_buffer(read_size);
        bool success = async_reader->prepare_read(buf, read_size, disk_offset, (void*)(uintptr_t)pageID);
        if (success) {
            pending_io_map[pageID] = buf;  // O(1) direct mapping for fast lookup
        } else {
            release_io_buffer(buf);
        }

        return success;
    };

    // Lambda: Poll for completed I/Os
    auto poll_completions = [&]() {
        if (!_use_async_io || async_reader == nullptr) return;

        std::vector<void*> completed_user_data;
        int num_completed = async_reader->poll_completions(completed_user_data, max_IOs);

        for (auto user_data : completed_user_data) {
            uint32_t node_id = (uint32_t)(uintptr_t)user_data;

            // O(1) map lookup instead of O(N) queue search!
            auto it = pending_io_map.find(node_id);
            if (it != pending_io_map.end()) {
                completed_io_map[node_id] = it->second;//assign the pointer of the buffer to from pending_io_map to completed_io_map
                pending_io_map.erase(it);  // Remove from pending
            }
        }
    };

    // Lambda: Wait for completed I/Os (blocking)
    auto wait_completions = [&]() {
        if (!_use_async_io || async_reader == nullptr) return;
        if (pending_io_map.empty()) return;  // No pending I/O, don't block

        std::vector<void*> completed_user_data;
        int num_completed = async_reader->wait_completions(completed_user_data, max_IOs);

        for (auto user_data : completed_user_data) {
            uint32_t node_id = (uint32_t)(uintptr_t)user_data;

            // O(1) map lookup (same as poll_completions!)
            auto it = pending_io_map.find(node_id);
            if (it != pending_io_map.end()) {
                completed_io_map[node_id] = it->second;
                pending_io_map.erase(it);  // Remove from pending
            }
        }
    };

    // End initialization timing
    auto init_end = std::chrono::high_resolution_clock::now();
    init_us = std::chrono::duration<float, std::micro>(init_end - query_start).count();

    // Start navigation timing
    auto nav_start = std::chrono::high_resolution_clock::now();

    // Helper lambda: Convert original vector ID to nav_idx.
    // Returns -1 if the vector is not a nav node.
    //
    // Two modes depending on how the nav graph was built:
    //
    // [Page subsampling mode] _nav_selected_pages is non-empty.
    //   Only the top vector (offset 0) of each selected page is a nav node.
    //   nav_idx is the position of that page in _nav_selected_pages.
    //
    //   Lookup uses O(1) formula inversion instead of a hash map:
    //     Forward: selected_page[i] = i * (N-1) / (K-1)   (integer division)
    //     Inverse: approx_i = page_id * (K-1) / (N-1)     (integer division)
    //   Because integer division truncates, approx_i may be off by ±1.
    //   Checking the three positions {approx_i-1, approx_i, approx_i+1} is always
    //   sufficient to find the true index if it exists.
    //
    // [Normal mode] _nav_selected_pages is empty.
    //   Every page is sampled; nav_idx = page_id * samples_per_page + sample_index.
    //   The sampling order within a page is: top(0), last, second-to-last, ...
    //   Given offset_in_page, sample_index is recovered in O(1):
    //     offset=0 → sample_index=0; offset=k → sample_index=capacity-k.
    auto original_id_to_nav_idx = [&](uint32_t original_id) -> int32_t {
        uint32_t page_id = original_id / static_cast<uint32_t>(_megaNode_capacity);
        uint32_t offset_in_page = original_id % static_cast<uint32_t>(_megaNode_capacity);

        if (!_nav_selected_pages.empty()) {
            // Page subsampling: only the top vector (offset 0) of a selected page qualifies.
            if (offset_in_page != 0) return -1;

            // O(1) inverse: approx_i = page_id * (K-1) / (N-1).
            // Clamp to [0, K-1] to guard against edge cases at the last page.
            uint32_t K = static_cast<uint32_t>(_nav_selected_pages.size());
            uint32_t N = _nav_total_pages;
            uint32_t approx_i = (K <= 1) ? 0
                : static_cast<uint32_t>(std::min((uint64_t)page_id * (K - 1) / (N - 1), (uint64_t)(K - 1)));

            // Check ±1 neighbourhood to correct for integer-division truncation error.
            for (int d = -1; d <= 1; ++d) {
                int64_t idx = (int64_t)approx_i + d;
                if (idx >= 0 && idx < (int64_t)K && _nav_selected_pages[idx] == page_id)
                    return static_cast<int32_t>(idx);
            }
            return -1;
        }

        // Normal mode: O(1) using sampling pattern
        bool is_last_page = (page_id == _last_mega_node_index);
        uint32_t page_capacity = is_last_page ?
            static_cast<uint32_t>(_last_page_vector_count) :
            static_cast<uint32_t>(_megaNode_capacity);
        uint32_t samples_this_page = is_last_page ?
            static_cast<uint32_t>(_nav_last_page_sample_indices.size()) :
            _nav_samples_per_page;

        uint32_t sample_index;
        if (offset_in_page == 0) {
            sample_index = 0;
        } else {
            sample_index = page_capacity - offset_in_page;
        }

        if (sample_index >= samples_this_page) return -1;
        return static_cast<int32_t>(page_id * _nav_samples_per_page + sample_index);
    };

    // Convergence detection state — declared here so it's accessible in both the
    // seeding phase and the main search loop.
    const uint32_t conv_rank = (stable_rank_threshold > 0) ? stable_rank_threshold : static_cast<uint32_t>(k_search);
    uint32_t prev_nth_id = UINT32_MAX;
    bool convergence_phase_start = false;

    ///MARK: Unified search starts here
    if (_nav_graph_loaded) {
        // Phase 1: Navigation graph search using PQ distances
        NeighborPriorityQueue &nav_retset = query_scratch->nav_retset;  // Use from scratch
        nav_retset.reserve(nav_L);
        memcpy(pq_coord_scratch, _cached_pq_buff + _nav_medoid_original_id * _n_chunks, _each_pq_space);
        diskann::pq_dist_lookup(pq_coord_scratch, 1, _n_chunks, pq_dists, dist_scratch);
        nav_retset.insert(Neighbor(_nav_medoid_original_id, dist_scratch[0]));

        // Greedy best-first search on navigation graph with PQ distances
        while (nav_retset.has_unexpanded_node()) {
            auto current = nav_retset.closest_unexpanded();
            uint32_t current_original_id = current.id;

            // Convert to nav_idx to look up neighbors
            int32_t nav_idx = original_id_to_nav_idx(current_original_id);
            if (nav_idx < 0 || static_cast<uint32_t>(nav_idx) >= _nav_num_points) {
                continue;  // Not a sampled vector or out of bounds
            }

            // Get navigation neighbors (stored as original vector IDs)
            uint32_t* nav_neighbors = _nav_nbrs + nav_idx * _nav_uniform_degree;

            // Batch compute PQ distances for all neighbors
            for (uint32_t i = 0; i < _nav_uniform_degree; i++) {
                memcpy(pq_coord_scratch + i * _each_pq_space,
                       _cached_pq_buff + nav_neighbors[i] * _n_chunks, _each_pq_space);
            }
            diskann::pq_dist_lookup(pq_coord_scratch, _nav_uniform_degree, _n_chunks, pq_dists, dist_scratch);

            // Insert neighbors into nav_retset
            for (uint32_t i = 0; i < _nav_uniform_degree; i++) {
                nav_retset.insert(Neighbor(nav_neighbors[i], dist_scratch[i]));
            }
        }

        // Phase 1.5: Nav-to-Disk Transition
        // After nav graph search converges, expand all unique pages from nav_retset
        // to find better entry points for disk search. For each page, compute PQ
        // distances for ALL vectors (not just sampled ones) and insert into retset.
        // seededPages tracks which pages have been processed here (separate from visitedPages
        // which is used for the main disk search so it remains clean).
        // Collect unique page IDs from nav_retset first.
        seededPages.reserve(nav_L);
        std::vector<uint32_t> unique_seed_pages;
        unique_seed_pages.reserve(nav_L);
        for (size_t i = 0; i < nav_retset.size(); i++) {
            uint32_t pageID = nav_retset[i].id / _megaNode_capacity;
            if (seededPages.insert(pageID).second)
                unique_seed_pages.push_back(pageID);
        }

        auto nav_expand_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < unique_seed_pages.size(); i++) {
            uint32_t pageID = unique_seed_pages[i];

            // Before seeding the last page, snapshot the k-th rank as prev_nth_id.
            // This gives the main search loop an immediate convergence signal on its
            // first iteration without needing a warm-up round.
            if (use_look_ahead_search && i == unique_seed_pages.size() - 1 && retset.size() >= conv_rank) {
                prev_nth_id = retset[conv_rank - 1].id;
            }

            uint32_t first_vec_ID = pageID * _megaNode_capacity;
            uint32_t num_nodes = (pageID == _last_mega_node_index) ?
                _last_page_vector_count : _megaNode_capacity;

            memcpy(pq_coord_scratch, _cached_pq_buff + first_vec_ID * _n_chunks,
                   num_nodes * _each_pq_space);
            diskann::pq_dist_lookup(pq_coord_scratch, num_nodes, _n_chunks, pq_dists, dist_scratch);

            for (uint32_t k = 0; k < num_nodes; k++) {
                retset.insert(Neighbor(first_vec_ID + k, dist_scratch[k]));
            }
        }

        // After all seeding: if the k-th rank didn't change after the last page,
        // the search has already converged during seeding.
        if (use_look_ahead_search && prev_nth_id != UINT32_MAX && retset.size() >= conv_rank) {
            if (retset[conv_rank - 1].id == prev_nth_id)
                convergence_phase_start = true;
        }

        // Note: visitedPages is NOT cleared here — it is untouched during Phase 1.5 so the
        // main disk search starts with an empty visitedPages (seededPages tracks Phase 1.5 state).
        auto nav_expand_end = std::chrono::high_resolution_clock::now();
        float nav_expand_us = std::chrono::duration<float, std::micro>(nav_expand_end - nav_expand_start).count();
    }
    else {
        // Use medoid of the base disk index graph when nav graph not loaded
        uint32_t pageID = _medoids[0];
        uint32_t medoid_vec_ID = static_cast<uint32_t>(pageID * _megaNode_capacity);
        memcpy(pq_coord_scratch, _cached_pq_buff + medoid_vec_ID * _n_chunks, _each_pq_space);
        diskann::pq_dist_lookup(pq_coord_scratch, 1, _n_chunks, pq_dists, dist_scratch);
        retset.insert(Neighbor(medoid_vec_ID, dist_scratch[0]));
    }

    // End navigation timing
    auto nav_end = std::chrono::high_resolution_clock::now();
    nav_us = std::chrono::duration<float, std::micro>(nav_end - nav_start).count();

    // Timing variables for I/O tracking
    auto io_start = std::chrono::high_resolution_clock::now();
    auto io_end = io_start;

    // ========================================
    // Look-ahead search state
    // ========================================
    // Enable look-ahead mode only if explicitly requested AND there are cached pages
    bool look_ahead_enabled = use_look_ahead_search && (_num_cached_pages > 0);
    bool look_ahead_mode = look_ahead_enabled;  // Start in look-ahead if enabled
    uint32_t first_skipped_node_id = UINT32_MAX;  // Track first skipped uncached node
    size_t current_beam_width = search_beam_width;
    bool convergence_beam_initialized = false;  // true after first convergence round sets the burst

    // Page-based search loop (using Linux AIO like PageANN)
    // Convergence at L: stop when the top-L entries are all expanded.
    // Entries L..2L-1 serve as overflow for Step 3 cached page processing during IO wait.
    while (retset.current_position() < l_search)
    {
        pages_to_read.clear();
        cached_pages.clear();

        // ========================================
        // PRE-ROUND PERSISTENCE CHECK
        // ========================================
        // Determine search mode for this round:
        // If first_skipped is known, scan top persistence_window_width unvisited nodes.
        // If found (prominent) → normal mode; not found → look-ahead mode.
        // Only relevant when convergence has not been reached yet.
        // After convergence: force normal mode and grow beam each round.
        if (convergence_phase_start) {
            look_ahead_mode = false;
            if (!convergence_beam_initialized) {
                // First convergence round: spike to L * beamwidth_spike_ratio, capped at l_search
                current_beam_width = std::min(static_cast<size_t>(l_search * beamwidth_spike_ratio), l_search);
                convergence_beam_initialized = true;
            } else {
                // Subsequent rounds: decay by beam_decay_ratio, floor at search_beam_width
                current_beam_width = std::max(static_cast<size_t>(current_beam_width * beam_decay_ratio), (size_t)search_beam_width);
            }
        } else if (look_ahead_enabled) {
            if (first_skipped_node_id != UINT32_MAX && persistence_window_width > 0) {
                bool found_first_skipped = false;
                uint32_t scanned = 0;
                for (size_t i = retset.current_position(); i < retset.size() && scanned < persistence_window_width; i++) {
                    const Neighbor& nbr = retset[i];
                    if (nbr.expanded) continue;
                    uint32_t pageID = static_cast<uint32_t>(nbr.id / _megaNode_capacity);
                    if (visitedPages.find(pageID) != visitedPages.end()) continue;
                    scanned++;
                    if (nbr.id == first_skipped_node_id) {
                        found_first_skipped = true;
                        break;
                    }
                }
                look_ahead_mode = !found_first_skipped;
            } else {
                look_ahead_mode = true;  // No skipped node tracked yet → default look-ahead
            }
        }

        // ========================================
        // SELECTION
        // ========================================
        uint32_t new_first_skipped = UINT32_MAX;

        if (look_ahead_mode) {
            // Look-ahead mode: collect cached pages only up to current_beam_width;
            // scan full retset (including overflow area) to maximise cached hits.
            // Track the first uncached page encountered as first_skipped_node_id.
            for (size_t i = retset.current_position(); i < retset.size(); i++) {
                Neighbor& nbr = retset[i];
                if (nbr.expanded) continue;
                uint32_t pageID = static_cast<uint32_t>(nbr.id / _megaNode_capacity);
                if (visitedPages.find(pageID) != visitedPages.end()) {
                    nbr.expanded = true;
                    continue;
                }
                if (is_page_cached(pageID)) {
                    nbr.expanded = true;
                    visitedPages.insert(pageID);
                    cached_pages.push_back(pageID);
                    if (cached_pages.size() >= current_beam_width) break;
                } else {
                    // Uncached disk page: track first occurrence as first_skipped
                    if (new_first_skipped == UINT32_MAX) new_first_skipped = nbr.id;
                    // Continue scanning for more cached pages
                }
            }
            first_skipped_node_id = new_first_skipped;
        } else {
            // Normal mode: collect cached and disk pages up to current_beam_width.
            // Post-convergence: current_beam_width grows each round (dynamic beamwidth).
            // Scan only within [cur, l_search) — overflow [l_search, capacity) is reserved
            // for cached pipeline overlap in Step 3.
            for (size_t i = retset.current_position(); i < l_search; i++) {
                Neighbor& nbr = retset[i];
                if (nbr.expanded) continue;
                uint32_t pageID = static_cast<uint32_t>(nbr.id / _megaNode_capacity);
                if (visitedPages.find(pageID) != visitedPages.end()) {
                    nbr.expanded = true;
                    continue;
                }
                nbr.expanded = true;
                visitedPages.insert(pageID);
                if (is_page_cached(pageID)) {
                    cached_pages.push_back(pageID);
                } else {
                    pages_to_read.push_back(pageID);
                }
                if (this->_count_visited_megaNodes) {
                    reinterpret_cast<std::atomic<uint32_t>&>(this->_mega_node_visit_counter[pageID].second).fetch_add(1);
                }
                if (cached_pages.size() + pages_to_read.size() >= current_beam_width) break;
            }
            // Track first skipped uncached node for the persistence window check next round.
            // Two conditions make this scan unnecessary: 1. look_ahead_enabled == false: look-ahead is fully disabled, so
            //   2. convergence_phase_start == true: convergence has been detected, so the persistence window check is bypassed for all remaining rounds and
            if (look_ahead_enabled && !convergence_phase_start) {
                for (size_t i = retset.current_position(); i < retset.size(); i++) {
                    const Neighbor& nbr = retset[i];
                    if (nbr.expanded) continue;
                    uint32_t pageID = static_cast<uint32_t>(nbr.id / _megaNode_capacity);
                    if (visitedPages.find(pageID) != visitedPages.end()) continue;
                    if (!is_page_cached(pageID)) {
                        new_first_skipped = nbr.id;
                        break;
                    }
                }
                first_skipped_node_id = new_first_skipped;
            }
        }

        // Update position to next unexpanded
        retset.update_cur_to_next_unexpanded();

        // ========================================
        // I/O AND PROCESSING
        // ========================================
        if (!pages_to_read.empty()) {
            if (use_pipeline) {
                // ======================================================
                // CASE 1 PIPELINE ON: Submit I/O first, overlap CPU with I/O wait
                // ======================================================
                io_start = std::chrono::high_resolution_clock::now();

                // 1. Submit async I/O for all uncached pages
                for (auto pageID : pages_to_read) {
                    submit_async_read(pageID);
                }
                int submitted = async_reader->submit_batch();
                if (stats != nullptr) stats->n_ios += submitted;
                num_ios += submitted;
                auto io_submit_end = std::chrono::high_resolution_clock::now();
                io_submit_us += std::chrono::duration<float, std::micro>(io_submit_end - io_start).count();

                // 2. Process selected cached pages during I/O wait (full vectors + neighbors)
                auto cached_proc_start = std::chrono::high_resolution_clock::now();
                n_cache_hits += static_cast<uint32_t>(cached_pages.size());
                for (auto pageID : cached_pages) {
                    char* page_data = get_cached_page_ptr(pageID);
                    T* page_vectors_buf = reinterpret_cast<T*>(page_data);
                    char* nhood_buf = page_data + _all_vectors_space_per_pageNode;
                    process_pageNode(pageID, page_vectors_buf, nhood_buf);
                }
                auto cached_proc_end = std::chrono::high_resolution_clock::now();
                cached_proc_us += std::chrono::duration<float, std::micro>(cached_proc_end - cached_proc_start).count();

                poll_completions();

                // 3. Process extra cached pages from retset (overlap with I/O wait)
                if (pending_io_map.size() > 0) {
                    size_t num_processed = 0;
                    size_t curr_pos = retset.current_position();
                    while (curr_pos < retset.size()) {
                        Neighbor& nbr = retset[curr_pos];
                        curr_pos++;
                        if (nbr.expanded) continue;

                        uint32_t pageID = static_cast<uint32_t>(nbr.id / _megaNode_capacity);
                        if (visitedPages.find(pageID) != visitedPages.end()) {
                            nbr.expanded = true;
                            continue;
                        }

                        if (is_page_cached(pageID)) {
                            nbr.expanded = true;
                            visitedPages.insert(pageID);
                            char* page_data = get_cached_page_ptr(pageID);
                            T* page_vectors_buf = reinterpret_cast<T*>(page_data);
                            char* nhood_buf = page_data + _all_vectors_space_per_pageNode;
                            process_pageNode(pageID, page_vectors_buf, nhood_buf);
                            num_processed++;
                            // Note: _count_visited_megaNodes is not incremented here because
                            // page-frequency profiling runs with caching disabled, so this
                            // pipeline-overlap path is never reached during profiling.

                            if (num_processed % 2 == 0) {
                                poll_completions();
                                if (pending_io_map.size() == 0) break;
                            }
                        }
                    }
                    n_cache_hits += static_cast<uint32_t>(num_processed);
                }

                // 4. Process deferred uncached vectors from previous hops (overlap with I/O wait)
                if (pending_io_map.size() > 0) {
                    size_t deferred_uncached_processed = 0;
                    for (size_t i = 0; i < deferred_proc_vec_uncached_page_ids.size(); ++i) {
                        uint32_t pageID = deferred_proc_vec_uncached_page_ids[i];
                        char *page_data_buf = deferred_proc_uncached_vec_buffer.data() + i * _all_vectors_space_per_pageNode;
                        T* page_vectors_buf = reinterpret_cast<T*>(page_data_buf);
                        process_pageNode(pageID, page_vectors_buf, nullptr);
                        deferred_uncached_processed++;

                        if (deferred_uncached_processed % 10 == 0) {
                            poll_completions();
                            if (pending_io_map.size() == 0) break;
                        }
                    }
                    size_t remaining = deferred_proc_vec_uncached_page_ids.size() - deferred_uncached_processed;
                    if (remaining > 0) {
                        memmove(deferred_proc_uncached_vec_buffer.data(),
                                deferred_proc_uncached_vec_buffer.data() + deferred_uncached_processed * _all_vectors_space_per_pageNode,
                                remaining * _all_vectors_space_per_pageNode);
                    }
                    deferred_proc_vec_uncached_page_ids.erase(
                        deferred_proc_vec_uncached_page_ids.begin(),
                        deferred_proc_vec_uncached_page_ids.begin() + deferred_uncached_processed);
                }

                // 5. Process deferred cached vectors from previous hops (overlap with I/O wait)
                if (pending_io_map.size() > 0) {
                    size_t deferred_cached_processed = 0;
                    for (auto pageID : deferred_proc_vec_cached_page_ids) {
                        char* page_data = get_cached_page_ptr(pageID);
                        T* page_vectors_buf = reinterpret_cast<T*>(page_data);
                        process_pageNode(pageID, page_vectors_buf, nullptr);
                        deferred_cached_processed++;

                        if (deferred_cached_processed % 10 == 0) {
                            poll_completions();
                            if (pending_io_map.size() == 0) break;
                        }
                    }
                    deferred_proc_vec_cached_page_ids.erase(
                        deferred_proc_vec_cached_page_ids.begin(),
                        deferred_proc_vec_cached_page_ids.begin() + deferred_cached_processed);
                }

                // 6. Wait for any remaining I/O
                auto io_wait_start = std::chrono::high_resolution_clock::now();
                while (pending_io_map.size() > 0) wait_completions();
                auto io_wait_end = std::chrono::high_resolution_clock::now();
                tail_io_us += std::chrono::duration<float, std::micro>(io_wait_end - io_wait_start).count();
                io_flight_us += std::chrono::duration<float, std::micro>(io_wait_end - io_start).count();

                // 7. Process disk pages: defer full vectors to next iteration, process neighbors now
                auto uncached_proc_start = std::chrono::high_resolution_clock::now();
                for (auto it = completed_io_map.begin(); it != completed_io_map.end(); ) {
                    uint32_t pageID = it->first;
                    char *buffer = it->second;

                    memcpy(deferred_proc_uncached_vec_buffer.data() + deferred_proc_vec_uncached_page_ids.size() * _all_vectors_space_per_pageNode,
                           buffer, _all_vectors_space_per_pageNode);
                    deferred_proc_vec_uncached_page_ids.push_back(pageID);
                    process_pageNode(pageID, nullptr, buffer + _all_vectors_space_per_pageNode);

                    release_io_buffer(buffer);
                    it = completed_io_map.erase(it);
                }
                auto uncached_proc_end = std::chrono::high_resolution_clock::now();
                uncached_proc_us += std::chrono::duration<float, std::micro>(uncached_proc_end - uncached_proc_start).count();

            } else {
                // ======================================================
                // CASE 1 PIPELINE OFF: Process cached first, then I/O, then uncached
                // No overlap — each phase is fully separate for clean latency measurement
                // ======================================================

                // 1. Process selected cached pages BEFORE submitting I/O
                auto cached_proc_start = std::chrono::high_resolution_clock::now();
                n_cache_hits += static_cast<uint32_t>(cached_pages.size());
                for (auto pageID : cached_pages) {
                    char* page_data = get_cached_page_ptr(pageID);
                    T* page_vectors_buf = reinterpret_cast<T*>(page_data);
                    char* nhood_buf = page_data + _all_vectors_space_per_pageNode;
                    process_pageNode(pageID, page_vectors_buf, nhood_buf);
                }
                auto cached_proc_end = std::chrono::high_resolution_clock::now();
                cached_proc_us += std::chrono::duration<float, std::micro>(cached_proc_end - cached_proc_start).count();

                // 2. Submit async I/O for uncached pages
                io_start = std::chrono::high_resolution_clock::now();
                for (auto pageID : pages_to_read) {
                    submit_async_read(pageID);
                }
                int submitted = async_reader->submit_batch();
                if (stats != nullptr) stats->n_ios += submitted;
                num_ios += submitted;
                auto io_submit_end = std::chrono::high_resolution_clock::now();
                io_submit_us += std::chrono::duration<float, std::micro>(io_submit_end - io_start).count();

                // 3. Wait for I/O completion
                auto io_wait_start = std::chrono::high_resolution_clock::now();
                while (pending_io_map.size() > 0) wait_completions();
                auto io_wait_end = std::chrono::high_resolution_clock::now();
                tail_io_us += std::chrono::duration<float, std::micro>(io_wait_end - io_wait_start).count();
                io_flight_us += std::chrono::duration<float, std::micro>(io_wait_end - io_start).count();

                // 4. Process disk pages immediately (full vectors + neighbors)
                auto uncached_proc_start = std::chrono::high_resolution_clock::now();
                for (auto it = completed_io_map.begin(); it != completed_io_map.end(); ) {
                    uint32_t pageID = it->first;
                    char *buffer = it->second;
                    process_pageNode(pageID, reinterpret_cast<T*>(buffer), buffer + _all_vectors_space_per_pageNode);
                    release_io_buffer(buffer);
                    it = completed_io_map.erase(it);
                }
                auto uncached_proc_end = std::chrono::high_resolution_clock::now();
                uncached_proc_us += std::chrono::duration<float, std::micro>(uncached_proc_end - uncached_proc_start).count();
            }

        } else {
            // CASE 2: ONLY CACHED PAGES
            // Pipeline ON:  process neighbors only, defer full vectors to next iteration's I/O wait
            // Pipeline OFF: process neighbors and full vectors immediately
            auto cached_proc_start = std::chrono::high_resolution_clock::now();
            n_cache_hits += static_cast<uint32_t>(cached_pages.size());
            for (auto pageID : cached_pages) {
                char* page_data = get_cached_page_ptr(pageID);
                T* page_vectors_buf = reinterpret_cast<T*>(page_data);
                char* nhood_buf = page_data + _all_vectors_space_per_pageNode;
                if (use_pipeline) {
                    process_pageNode(pageID, nullptr, nhood_buf);
                    deferred_proc_vec_cached_page_ids.push_back(pageID);
                } else {
                    process_pageNode(pageID, page_vectors_buf, nhood_buf);
                }
            }
            auto cached_proc_end = std::chrono::high_resolution_clock::now();
            cached_proc_us += std::chrono::duration<float, std::micro>(cached_proc_end - cached_proc_start).count();
        }

        // Convergence check: update prev_nth_id only while not yet converged
        if (look_ahead_enabled && !convergence_phase_start && retset.size() >= conv_rank) {
            uint32_t curr_nth_id = retset[conv_rank - 1].id;
            if (curr_nth_id == prev_nth_id)
                convergence_phase_start = true;
            else
                prev_nth_id = curr_nth_id;
        }

        hops++;
        if (stats != nullptr) {
            stats->n_hops++;
        }
    }// search converged


    // FINAL CLEANUP: Process remaining deferred full vectors after search converges
    // 1. Process deferred cached pages (full vectors only)
    for (auto pageID : deferred_proc_vec_cached_page_ids) {
        char* page_data = get_cached_page_ptr(pageID);
        T* page_vectors_buf = reinterpret_cast<T*>(page_data);
        process_pageNode(pageID, page_vectors_buf, nullptr);
    }

    // 2. Process deferred uncached pages (full vectors only)
    for (size_t i = 0; i < deferred_proc_vec_uncached_page_ids.size(); ++i)
    {
        uint32_t pageID = deferred_proc_vec_uncached_page_ids[i];
        char *page_data_buf = deferred_proc_uncached_vec_buffer.data() + i * _all_vectors_space_per_pageNode;
        T* page_vectors_buf = reinterpret_cast<T*>(page_data_buf);
        process_pageNode(pageID, page_vectors_buf, nullptr);
    }

    // Copy top-k results to output
    for (uint64_t i = 0; i < k_search; i++)
    {
        res_ids[i] = full_ret_queue[i].id;
        if (res_dists != nullptr)
        {
            res_dists[i] = full_ret_queue[i].distance;
        }
    }

    // Save all timing statistics
    if (stats != nullptr)
    {
        auto query_end = std::chrono::high_resolution_clock::now();
        float total_us = std::chrono::duration<float, std::micro>(query_end - query_start).count();

        stats->total_us = total_us;
        stats->init_us = init_us;
        stats->nav_us = nav_us;
        stats->tail_io_us = tail_io_us;
        stats->io_flight_us = io_flight_us;
        stats->io_submit_us = io_submit_us;
        stats->cpu_us = cached_proc_us + uncached_proc_us;
        stats->cached_full_data_during_io_us = cached_proc_us;
        stats->uncached_full_data_during_io_us = uncached_proc_us;
        stats->n_cache_hits = n_cache_hits;
        stats->n_cmps = full_precision_cmps;
        stats->n_non_hub_requested = non_hub_requested;

        // Count distinct IO pages whose vectors appear in the final retset.
        // A page is an "IO page" if it was visited but not cached (i.e., read from disk).
        // Overflow retset entries (positions >= l_search) that are uncached were never read —
        // visitedPages won't contain them, so the check is naturally correct.
        uint32_t useful = 0;
        tsl::robin_set<uint32_t> seen_pids;
        for (size_t i = 0; i < retset.size(); i++)
        {
            uint32_t pid = static_cast<uint32_t>(retset[i].id / _megaNode_capacity);
            if (seen_pids.insert(pid).second)  // first time seeing this page
            {
                if (!is_page_cached(pid) && visitedPages.count(pid))
                    useful++;
            }
        }
        stats->n_useful_ios = useful;
    }
}

///MARK: // instantiations
template class PQFlashIndexLAANN<uint8_t>;
template class PQFlashIndexLAANN<int8_t>;
template class PQFlashIndexLAANN<float>;
template class PQFlashIndexLAANN<uint8_t, uint16_t>;
template class PQFlashIndexLAANN<int8_t, uint16_t>;
template class PQFlashIndexLAANN<float, uint16_t>;

} // namespace diskann
