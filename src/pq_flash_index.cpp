// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Page-based Index Search Engine
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"
#include <unordered_map>
#include <nmmintrin.h>

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#ifdef USE_BING_INFRA
#warning "USE_BING_INFRA is enabled"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &dataReader, diskann::Metric m)
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

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);

    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    // Manual cleanup: Deleting the raw pointers
    if (_sample_nodes_IDs_in_LSH != nullptr){
        delete[] _sample_nodes_IDs_in_LSH;
    }

    if (_projectionMatrix != nullptr){
        delete[] _projectionMatrix;
    }

    if (_cached_pq_buff != nullptr){
        delete[] _cached_pq_buff;
    }

    _buckets.clear();  // Clear the map after manual deletion
    
    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->dataReader->deregister_all_threads();
        dataReader->close();
        diskann::cout << "Cleared scratch" << std::endl;
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
    diskann::cout << "Destroyed PQ Flash Index Object" << std::endl;

}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    //first page is for the meta data so we increment it by 1
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve, _nnodes_per_sector);//each data includes a SSDQueryScratch and one IOContext
            //this->topoReader->register_thread();//NOTE: there is warning here -- called by both WindowsAlignedFileReader and load
            this->dataReader->register_thread();
///NOTE: no need to change the AlignedReader file cuz ctx_map belongs to each reader object
            //data->topo_ctx = this->topoReader->get_ctx();
            data->data_ctx = this->dataReader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}


template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<uint32_t *> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
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

        ///NOTE: this only used for use_medoids_data_as_centroids --- which just need to first node of the page to represent the medoid page
        if (coord_buffers[i] != nullptr)
        {
            //this cast char * to T*
            T *node_coords = offset_to_node_coords(node_buf);
            //_disk_bytes_per_point was calculated as dim * sizof(T)
            //the third parameter: the number of bytes to copy from the source to the destination.
            memcpy(coord_buffers[i], node_coords, _nnodes_per_sector * _disk_bytes_per_point);
        }
    }

    aligned_free(buf);

    return retval;
}


template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_page_nbrs(const std::vector<uint32_t> &page_ids, std::vector<std::vector<uint32_t>*> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(page_ids.size(), true);

    char *buf = nullptr;//this is used to store all the sector to read into mem
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, page_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
    std::memset(buf, 0, page_ids.size() * num_sectors * defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < page_ids.size(); ++i)
    {
        auto page_id = page_ids[i];
        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        ///MARK: first page is for the meta data so we increment it by 1
        read.offset = (page_id + 1) * defaults::SECTOR_LEN;
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

        char *node_buf = (char *)read_reqs[i].buf;
        uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(node_buf + _nnodes_per_sector * _disk_bytes_per_point);
        uint32_t num_cached_nbrs_32 = (uint32_t)num_cached_nbrs;
        uint32_t* node_nhood = reinterpret_cast<uint32_t*>(node_buf + _nnodes_per_sector * _disk_bytes_per_point + 2 * sizeof(uint16_t));
        // Correctly write into the vector
        nbr_buffers[i]->assign(node_nhood, node_nhood + num_cached_nbrs_32);
    }

    aligned_free(buf);
    return retval;
}

//in this function, the coord_buffers store all nodes' values within the page instead of just the representative one
template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_page_all_nodes(const std::vector<uint32_t> &page_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<uint32_t *> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(page_ids.size(), true);

    char *buf = nullptr;//this is used to store all the sector to read into mem
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, page_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
  
    // create read requests
    for (size_t i = 0; i < page_ids.size(); ++i)
    {
        auto page_id = page_ids[i];
        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
       //first page is for the meta data so we increment it by 1
        read.offset = (page_id + 1) * defaults::SECTOR_LEN;
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

        char *node_buf = (char *)read_reqs[i].buf;

        if (coord_buffers[i] != nullptr)
        {
            //this just cast char * to T*
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point * _nnodes_per_sector);
        }

        if (nbr_buffers[i] != nullptr)
        {
            uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(node_buf + _nnodes_per_sector * _disk_bytes_per_point);
            uint32_t num_cached_nbrs_32 = (uint32_t)num_cached_nbrs;
            nbr_buffers[i][0] = num_cached_nbrs_32;
            uint32_t* node_nhood = reinterpret_cast<uint32_t*>(node_buf + _nnodes_per_sector * _disk_bytes_per_point + 2 * sizeof(uint16_t));
            std::memcpy(&nbr_buffers[i][1], node_nhood, num_cached_nbrs_32 * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &page_list)
{
   
    diskann::cout << "Loading frequently visited pages into memory.." << std::flush;
    size_t num_cached_pages = page_list.size();

    ///MARK: 1. Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_pages * _nnodes_per_sector * _data_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));//this 8 is hard-coded for alignment for optimization purpose
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    ///MARK: 2. Allocate space for neighborhood cache
    //the first 32bits of each "block" is nnbrs
    _nhood_cache_buf = new uint32_t[num_cached_pages * (_max_degree + 1)]; //all nodes within a page share a common set of nbrs
    memset(_nhood_cache_buf, 0, num_cached_pages * (_max_degree + 1) * sizeof(uint32_t));

    ///MARK: current version: enable cache only if all pq are loaded
    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_pages, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_pages, (block + 1) * BLOCK_SIZE);

        std::vector<uint32_t> pages_to_read;
        std::vector<uint32_t> idx_in_buff;
        ///MARK: store all nodes within the page as 1D vector instead of 2D vector --- just index by page with offset of _nnodes_per_sector * _data_dim
        std::vector<T *> coord_buffers; 
        std::vector<uint32_t *> nbr_buffers; //the vector of the pointer to the neigbrs of each page in order
        for (size_t idx = start_idx; idx < end_idx; idx++)
        {
            pages_to_read.push_back(page_list[idx]);
            idx_in_buff.push_back(static_cast<uint32_t>(idx));////we need offset array
            coord_buffers.push_back(_coord_cache_buf + idx * _nnodes_per_sector * _data_dim);//save into coord_buffers the pointers -- namely, where to store these data when reading
            //coord_buffers.push_back(nullptr);
            nbr_buffers.push_back(_nhood_cache_buf + idx * (_max_degree + 1));
        }

        ///MARK:  make and use a new function
        auto read_status = read_page_all_nodes(pages_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _cached_page_idx_map.insert(std::make_pair(pages_to_read[i], idx_in_buff[i]));
            }
        }
    }

    diskann::cout << "Done of loading the most frequently visited pages." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint32_t num_pages_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &page_list, bool use_hash_routing)
{
#endif
    if(!_all_pq_cached){
        std::cout << "Cannot cache most frequently visited pages unless full PQ data are loaded in RAM." <<std::endl;
        return;
    }

    if (num_pages_to_cache >= this->_num_pages)
    {
        // for small num_points and big num_pages_to_cache, use below way to get the page_list quickly
        page_list.resize(this->_num_pages);
        for (uint32_t i = 0; i < this->_num_pages; ++i)
        {
            page_list[i] = i;
        }
        return;
    }

    bool load_from_top_pages_file = false;
    if(load_from_top_pages_file){
        std::ifstream top_pages_reader("top_pages.bin", std::ios::binary);
        if (!top_pages_reader) {
            throw std::runtime_error("Failed to open top_pages.bin file for reading.");
        }

        uint32_t num_pages = 0;
        top_pages_reader.read(reinterpret_cast<char*>(&num_pages), sizeof(uint32_t));
        if (!top_pages_reader) {
            throw std::runtime_error("Failed to read the number of pages from top_pages.bin.");
        }

        std::vector<uint32_t> top_pages(num_pages);
        top_pages_reader.read(reinterpret_cast<char*>(top_pages.data()), num_pages * sizeof(uint32_t));
        top_pages_reader.close();
        std::cout << "Loaded top " << num_pages << " top pages from file.\n";

        // Use assign for more concise and efficient copying
        page_list.assign(top_pages.begin(), top_pages.begin() + num_pages_to_cache);
        return;
    }
    
    this->_count_visited_pages = true;
    this->_page_visit_counter.clear();
    this->_page_visit_counter.resize(this->_num_pages);
    for (uint32_t i = 0; i < _page_visit_counter.size(); i++)
    {
        this->_page_visit_counter[i].first = i;
        this->_page_visit_counter[i].second = 0;
    }

    ///MARK: might be useful for others
    // this->_getMostFrequentlyVisitedNodes = false;
    // if(this->_getMostFrequentlyVisitedNodes){
    //     this->_node_visit_counter.clear();
    //     this->_node_visit_counter.resize(this->_num_points);
    //     for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    //     {
    //         this->_node_visit_counter[i].first = i;
    //         this->_node_visit_counter[i].second = 0;
    //     }
    // }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }
    std::cout << "Done of loading sample files.\n";

    std::vector<float> query_result_dists(1 * sample_num);
    std::vector<uint64_t> query_result_ids(1 * sample_num);
    auto stats = new diskann::QueryStats[sample_num];

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
       
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        linux_page_search(samples + (i * sample_aligned_dim), 1, l_search, query_result_ids.data() + i * 1,
                         query_result_dists.data() + i * 1, beamwidth, false, stats, use_hash_routing);
    }
    
    std::cout << "Done of stats. Begin to sort.\n";

    std::sort(this->_page_visit_counter.begin(), _page_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    std::cout << "Done of sorting.\n";
    
    bool write_top_pages = false;
    if(write_top_pages){

        std::ofstream outFile("/mnt/c/Users/dxk230039/Desktop/top_pages.bin", std::ios::binary);
        if (!outFile.is_open()) {
            throw std::runtime_error("Failed to open file for writing top visited page IDs.");
        }
        
        // Write the total number of nodes to record at the beginning
        std::cout << "num_pages: " << this->_num_pages << std::endl;
        uint32_t num_pages_to_record_u32 = static_cast<uint32_t>(this->_num_pages);
        outFile.write(reinterpret_cast<const char*>(&num_pages_to_record_u32), sizeof(uint32_t));
        std::cout << "Write the ID of the top " << num_pages_to_record_u32 << " most frequently visited pages into file";
        
        for (uint32_t i = 0; i < num_pages_to_record_u32; i++) {
            uint32_t page_id = this->_page_visit_counter[i].first;
            outFile.write(reinterpret_cast<const char*>(&page_id), sizeof(uint32_t));
        }
        
        outFile.close();
    }

    page_list.clear();
    page_list.shrink_to_fit();
    page_list.reserve(num_pages_to_cache);
    for (uint32_t i = 0; i < num_pages_to_cache; i++)
    {
        page_list.push_back(this->_page_visit_counter[i].first);
    }

    this->_count_visited_pages = false;
    std::cout << "Put page ID to page list.\n";
    diskann::aligned_free(samples);
    std::cout << "Done with freeing sample files.\n";
}

///MARK: updated
template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    //_centroid_data is a pointer of float array
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    ///MARK: updated
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _nnodes_per_sector * _data_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _nnodes_per_sector * _data_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<uint32_t *> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_nnodes_per_sector * _data_dim]);//this is used to store the all nodes of the page
        nbr_bufs.push_back(nullptr);//won't be used for this function. but need to be changed for cache
    }

//this read_nodes only uses the medoid_bufs but nbr_bufs
    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);
    diskann::cout << "Finish reading " << _num_medoids << " medoid(s)" << std::endl;
    //MARK: centroids are same as medoids if there is no centroids file
    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {   
                for (uint32_t k = 0; k < _nnodes_per_sector; k++){
                    for (uint32_t i = 0; i < _data_dim; i++){
                        _centroid_data[cur_m * _nnodes_per_sector * _data_dim + k * _data_dim + i] = medoid_bufs[cur_m][k * _data_dim + i];
                    }
                }
                diskann::cout << "Finish loading " << _num_medoids << " medoid(s)" << std::endl;
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];//we need to explictly delete because it is created with new
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix, const std::string &pq_path_prefix, const bool use_hash_routing, const bool use_sampled_hash_routing, const uint32_t radius)
{
#endif
    _radius = radius;

    std::string pq_table_bin = (pq_path_prefix.empty() ? std::string(index_prefix) : pq_path_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_reorder_file = (pq_path_prefix.empty() ? std::string(index_prefix) : pq_path_prefix) + "_reorder_pq_compressed.bin";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, index_prefix, pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str(), use_hash_routing, use_sampled_hash_routing);
#else
    return load_from_separate_paths(num_threads, index_prefix, pq_table_bin.c_str(),
                                    pq_compressed_reorder_file.c_str(), use_hash_routing, use_sampled_hash_routing);
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath, const bool use_hash_routing, const bool use_sampled_hash_routing)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *pq_compressed_reorder_file_path, const bool use_hash_routing, const bool use_sampled_hash_routing)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_reorder_file = pq_compressed_reorder_file_path;
    std::string _disk_index_file = std::string(index_filepath) + ".index";
    std::string medoids_file = "";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    size_t nchunks_u64;
#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    this->_n_chunks = nchunks_u64; 

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file dataReader approach.
    diskann::cout << "If EXEC_ENV_OLS" << std::endl;
    dataReader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);//this is the number of meta info and the dimension of each data info
    READ_U32(index_metadata, nc);

    if (!index_metadata) {
        std::cerr << "Error opening file: " << _disk_index_file << std::endl;
        return -1;
    }

    uint64_t disk_nnodes;
    uint64_t disk_ndims;
    uint64_t page_size;

    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);
    READ_U64(index_metadata, page_size);

    this->_data_dim = disk_ndims;
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(disk_ndims, 8);
    this->_num_points = disk_nnodes;
    this->_n_chunks = nchunks_u64;
    diskann::cout << "#points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    size_t medoid_id_on_file, pq_ndims;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    READ_U64(index_metadata, pq_ndims);
    READ_U64(index_metadata, _max_degree);

    uint64_t theoretial_max_degree = (defaults::SECTOR_LEN - _nnodes_per_sector * _max_node_len - 2 * sizeof(uint16_t)) / sizeof(uint32_t);
    this->_num_pages = (disk_nnodes + _nnodes_per_sector - 1)/_nnodes_per_sector;

    if (_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }


    diskann::cout << "Disk-Index File Meta-data: " << std::endl;
    diskann::cout << "medoid's pageID: " << medoid_id_on_file << std::endl;
    diskann::cout << "num of pages: " << this->_num_pages << std::endl;
    diskann::cout << "initial PQ dim: " << pq_ndims << std::endl;
    diskann::cout << "PQ dim in use: " << _n_chunks << std::endl;
    diskann::cout << "# nodes per sector: " << _nnodes_per_sector << std::endl;
    diskann::cout << "max node len (bytes): " << _max_node_len << std::endl;
    diskann::cout << "theoretical max page-graph degree: " << theoretial_max_degree << std::endl;
    diskann::cout << "actual max page-graph degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

    // Load hash routing buckets
    if(use_hash_routing || use_sampled_hash_routing){
        std::string bucketsFile = std::string(index_filepath) + (use_sampled_hash_routing ? "_subset_buckets.bin" : "_buckets.bin");
        std::ifstream inFile(bucketsFile, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open hash buckets file for reading.");
        }

        // Read hash bucket metadata
        size_t bucketCount, num_sampled_nodes_in_lsh;
        int numProjections;
        inFile.read(reinterpret_cast<char*>(&bucketCount), sizeof(bucketCount));
        inFile.read(reinterpret_cast<char*>(&numProjections), sizeof(numProjections));
        inFile.read(reinterpret_cast<char*>(&num_sampled_nodes_in_lsh), sizeof(num_sampled_nodes_in_lsh));
        _numProjections = numProjections;

        size_t bucket_data_size = bucketCount * (sizeof(uint32_t) + sizeof(uint8_t)) + num_sampled_nodes_in_lsh * sizeof(uint32_t); 

        std::vector<char> fileBuffer(bucket_data_size);
        inFile.read(fileBuffer.data(), bucket_data_size);
        inFile.close();
        const char* ptr = fileBuffer.data();

        diskann::cout << "Successfully opened file: " << bucketsFile << std::endl;
        diskann::cout << "Number of buckets: " << bucketCount << std::endl;
        diskann::cout << "Number of projections: " << numProjections << std::endl;
        diskann::cout << "Number of hashed sampled nodes: " << num_sampled_nodes_in_lsh << std::endl;

        _buckets.clear();
        _buckets.reserve(bucketCount);

        _sample_nodes_IDs_in_LSH = new uint32_t[num_sampled_nodes_in_lsh];
        uint32_t* currentPtr = _sample_nodes_IDs_in_LSH;
        for (size_t i = 0; i < bucketCount; ++i) {
            uint32_t hash;
            uint8_t idCount;
            memcpy(&hash, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
            memcpy(&idCount, ptr, sizeof(uint8_t));
            ptr += sizeof(uint8_t);
            memcpy(currentPtr, ptr, idCount * sizeof(uint32_t));
            ptr += idCount * sizeof(uint32_t);
            _buckets[hash] = {currentPtr, idCount};
            currentPtr += idCount;

            if (i % 1000 == 0 || i == bucketCount - 1) {
                float progress = (100.0f * i) / bucketCount;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                        << progress << "% " << std::flush;
            }
        }
        std::cout << "\nBuckets map successfully loaded.\n";

        // Load projection matrix
        std::string projectionMatrixFile = std::string(index_filepath) + "_projection_matrix_file.bin";
        std::ifstream inFilePM(projectionMatrixFile, std::ios::binary);
        if (!inFilePM) {
            throw std::runtime_error("Failed to open projection matrix file for reading.");
        }

        int numProjections_from_PM, dimensions;
        inFilePM.read(reinterpret_cast<char*>(&numProjections_from_PM), sizeof(numProjections_from_PM));
        inFilePM.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));

        _projectionMatrix = new float[numProjections_from_PM * dimensions];
        inFilePM.read(reinterpret_cast<char*>(_projectionMatrix), numProjections_from_PM * dimensions * sizeof(float));
        inFilePM.close();
        std::cout << "Projection matrix successfully loaded.\n";

        diskann::cout << "Number of buckets: " << _buckets.size();
        diskann::cout << ", numProjections: " << _numProjections;
        diskann::cout << ", number of dimensions: " << dimensions << std::endl;
        std::cout << "Hash routing search radius: " << static_cast<size_t>(_radius) << std::endl;
    }

    // Load cached PQ data
    std::ifstream inPQFile(pq_compressed_reorder_file, std::ios::binary);
    if (!inPQFile.is_open()) {
        throw std::runtime_error("Failed to open reorder compressed PQ data file for reading.");
    }
    uint32_t points_num, num_pq_chunks;
    uint8_t useID_pqIdx_map_raw, useID_pqIdx_array_raw;
    
    // {
    //     std::ifstream status("/proc/self/status");
    //     std::string line;
    //     while (std::getline(status, line)) {
    //         if (line.rfind("VmRSS", 0) == 0 || line.rfind("VmSize", 0) == 0) {
    //             std::cout << "[" << "before loading PQ" << "] " << line << std::endl;
    //         }
    //     }
    // }

    inPQFile.read(reinterpret_cast<char*>(&points_num), sizeof(points_num));
    inPQFile.read(reinterpret_cast<char*>(&_num_pq_cached_nodes), sizeof(_num_pq_cached_nodes));
    inPQFile.read(reinterpret_cast<char*>(&num_pq_chunks), sizeof(num_pq_chunks));
    inPQFile.read(reinterpret_cast<char*>(&useID_pqIdx_map_raw), sizeof(useID_pqIdx_map_raw));
    inPQFile.read(reinterpret_cast<char*>(&useID_pqIdx_array_raw), sizeof(useID_pqIdx_array_raw));
    _useID_pqIdx_map = useID_pqIdx_map_raw != 0;
    _useID_pqIdx_array = useID_pqIdx_array_raw != 0;

    std::cout << "points_num: " << points_num << std::endl;
    std::cout << "_num_pq_cached_nodes: " << _num_pq_cached_nodes << std::endl;
    std::cout << "num_pq_chunks: " << num_pq_chunks << std::endl;
    std::cout << "_useID_pqIdx_map: " << (_useID_pqIdx_map ? "true" : "false") << std::endl;
    std::cout << "_useID_pqIdx_array: " << (_useID_pqIdx_array ? "true" : "false") << std::endl;

    if (_num_pq_cached_nodes > 0){
        _no_pq_cached = false;
        if (points_num == _num_pq_cached_nodes){
            _all_pq_cached = true;
            diskann::cout << "All PQ data are loaded to memory." << std::endl;
        }
        uint64_t buffer_size = static_cast<uint64_t>(_num_pq_cached_nodes) * num_pq_chunks;
        _cached_pq_buff = new uint8_t[buffer_size];
        inPQFile.read(reinterpret_cast<char*>(_cached_pq_buff), buffer_size * sizeof(uint8_t));
        diskann::cout << "PQ buffer size: " << buffer_size << "B" << std::endl;

        // {
        //     std::ifstream status("/proc/self/status");
        //     std::string line;
        //     while (std::getline(status, line)) {
        //         if (line.rfind("VmRSS", 0) == 0 || line.rfind("VmSize", 0) == 0) {
        //             std::cout << "[" << "after loading PQ values" << "] " << line << std::endl;
        //         }
        //     }
        // }

        if (_useID_pqIdx_map){
            std::vector<uint32_t> nodeID_pqIdx_array_tem(_num_pq_cached_nodes);
            inPQFile.read(reinterpret_cast<char*>(nodeID_pqIdx_array_tem.data()), _num_pq_cached_nodes * sizeof(uint32_t));
            _nodeID_pqIdx_map.reserve(_num_pq_cached_nodes);
            std::cout << "Bucket count after reserve: " << _nodeID_pqIdx_map.bucket_count() << std::endl;
            for(uint32_t i = 0; i < _num_pq_cached_nodes; i++){
                _nodeID_pqIdx_map[nodeID_pqIdx_array_tem[i]] = i;
            }
            std::cout << "_nodeID_pqIdx_map.size(): " << _nodeID_pqIdx_map.size() << std::endl;
            std::cout << "capacity(): " << _nodeID_pqIdx_map.bucket_count() << std::endl;
        }
        else if(_useID_pqIdx_array){
            _nodeID_pqIdx_arr.resize(points_num);
            inPQFile.read(reinterpret_cast<char*>(_nodeID_pqIdx_arr.data()), points_num * sizeof(uint32_t));
        }

        // {
        //     std::ifstream status("/proc/self/status");
        //     std::string line;
        //     while (std::getline(status, line)) {
        //         if (line.rfind("VmRSS", 0) == 0 || line.rfind("VmSize", 0) == 0) {
        //             std::cout << "[" << "after loading PQ map" << "] " << line << std::endl;
        //         }
        //     }
        // }
    }
    inPQFile.close();

#ifndef EXEC_ENV_OLS
    std::string index_fname(_disk_index_file);
    dataReader->open(index_fname);

    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    diskann::cout << "Index loading complete." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

// Wrapper: page_search without filter and IO limit
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::page_search(const T *query1, const uint64_t k_search, const uint64_t l_search, uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data, QueryStats *stats, const bool use_hash_routing)
{
    page_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, stats, use_hash_routing);
}

// Wrapper: page_search with filter
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::page_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats, const bool use_hash_routing)
{
    page_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, stats, use_hash_routing);
}

// Wrapper: page_search with IO limit
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::page_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats, const bool use_hash_routing)
{
    LabelT dummy_filter = 0;
    page_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, stats, use_hash_routing);
}

// Main page_search implementation with all parameters
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::page_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats, const bool use_hash_routing)
{
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__, __LINE__);

    // Get scratch space from thread pool
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &data_ctx = data->data_ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    query_scratch->reset();

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
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
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
            aligned_query_T[i] = query1[i];
        }
    }
    pq_query_scratch->initialize(this->_data_dim, aligned_query_T);

    // Buffers for node data and distance calculations
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // Precompute PQ distances
    _pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;
    tsl::robin_set<uint32_t> &expandedPages = query_scratch->expandedPages;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;
    uint32_t full_precision_cmps = 0;
    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();

    bool useMediod = true;

    uint64_t each_nbr_ID_PQ_space = (uint64_t)(sizeof(uint32_t) + _n_chunks * sizeof(uint8_t));
    uint64_t each_pq_space = (uint64_t)(_n_chunks * sizeof(uint8_t));
    std::size_t coords_stride = _nnodes_per_sector * _data_dim;
    std::size_t nhood_stride  = _max_degree + 1;
    uint64_t all_full_vector_space = _nnodes_per_sector * _max_node_len;

    // LSH-based entry point selection
    if (use_hash_routing){
        useMediod = false;

        // Normalize query vector
        float norm = 0.0f;
        for (size_t j = 0; j < _data_dim; ++j) {
            norm += query_float[j] * query_float[j];
        }
        norm = std::sqrt(norm);
        float norm_inv = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
        float* normalized_query = new float[_data_dim];
        for (size_t j = 0; j < _data_dim; ++j) {
            normalized_query[j] = query_float[j] * norm_inv;
        }

        // Compute LSH hash for query
        uint32_t query_hash = 0;
        for (int i = 0; i < _numProjections; ++i) {
            float dotProduct = 0.0;
            size_t offset = i * _data_dim;
            for (size_t j = 0; j < _data_dim; ++j) {
                dotProduct += normalized_query[j] * _projectionMatrix[offset + j];
            }
            if (dotProduct >= 0) {
                query_hash |= (1U << i); 
            }
        }
        delete[] normalized_query;

        // Generate neighboring hashes within Hamming radius
        tsl::robin_set<uint32_t> neighbor_hashes;
        neighbor_hashes.insert(query_hash);
        std::vector<uint32_t> temp_neighbors = {query_hash};

        for (int r = 1; r <= _radius; ++r) {
            std::vector<uint32_t> new_neighbors;
            for (uint32_t h : temp_neighbors) {
                for (int i = 0; i < _numProjections; ++i) {
                    uint32_t neighbor_hash = h ^ (1U << i);
                    if (neighbor_hashes.insert(neighbor_hash).second) {
                        new_neighbors.push_back(neighbor_hash);
                    }
                }
            }
            temp_neighbors = std::move(new_neighbors);
        }

        // Collect nodes from LSH buckets
        std::vector<uint32_t> closeNodes;
        uint8_t *pq_coord_scratch_buf = pq_coord_scratch;
        for (uint32_t hash : neighbor_hashes) {
            auto it = _buckets.find(hash);
            if (it != _buckets.end()) {
                uint32_t* buff_ptr = it->second.first;
                uint8_t num_IDs = it->second.second;
                uint32_t id;
                for (uint8_t i = 0; i < num_IDs; ++i) {
                    memcpy(&id, buff_ptr + i, sizeof(uint32_t));
                    closeNodes.push_back(id);

                    uint32_t pd_idx = id;
                    if(_useID_pqIdx_map){
                        pd_idx = _nodeID_pqIdx_map[id];
                    }else if (_useID_pqIdx_array){
                        pd_idx = _nodeID_pqIdx_arr[id];
                    }
                    memcpy(pq_coord_scratch_buf, _cached_pq_buff + pd_idx * _n_chunks, _n_chunks * sizeof(uint8_t));
                    pq_coord_scratch_buf += _n_chunks * sizeof(uint8_t);

                    if (closeNodes.size() >= defaults::MAX_GRAPH_DEGREE){
                        break;
                    }
                }
            }
            if (closeNodes.size() >= defaults::MAX_GRAPH_DEGREE){
                break;
            }
        }

        diskann::pq_dist_lookup(pq_coord_scratch, closeNodes.size(), _n_chunks, pq_dists, dist_scratch);

        for(size_t i = 0; i < closeNodes.size(); ++i){
            uint32_t nid = closeNodes[i];
            float dist = dist_scratch[i];
            Neighbor nn(nid, dist);
            retset.insert(nn);
        }

        stats->n_lsh_entry_points += static_cast<uint32_t>(retset.size());

        if (retset.size() == 0) {
            useMediod = true;
        }
    }

    // Medoid-based entry point selection
    if (useMediod){
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            uint32_t firstNodeID = static_cast<uint32_t>(_medoids[cur_m] * _nnodes_per_sector);
            for (uint8_t k = 0; k < _nnodes_per_sector; k++){
                float *node_disk_buf = _centroid_data + cur_m * _nnodes_per_sector * _data_dim + k * _data_dim;
                T *node_fp_coords = (T *)(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                uint32_t nbr_id = firstNodeID + k;
                full_retset.push_back(Neighbor(nbr_id, dist));

                if (dist < best_dist)
                {
                    best_medoid = nbr_id;
                    best_dist = dist;
                }
            }
            full_precision_cmps += (uint32_t)_nnodes_per_sector;
        }
        retset.insert(Neighbor(best_medoid, best_dist));
    }

    uint32_t hops = 0;
    uint32_t num_ios = 0;

    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, uint32_t>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // Beam search main loop
    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;

        // Select beam_width pages to expand
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            uint32_t pageID = static_cast<uint32_t>(nbr.id / _nnodes_per_sector);
            if (expandedPages.insert(pageID).second){
                num_seen++;
                auto iter = _cached_page_idx_map.find(pageID);
                if (iter != _cached_page_idx_map.end()){
                    cached_nhoods.emplace_back(pageID, iter->second);
                    if (stats != nullptr){
                        stats->n_cache_hits++;
                    }
                }
                else{
                    frontier.push_back(pageID);
                }

                if (this->_count_visited_pages)
                {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_page_visit_counter[pageID].second).fetch_add(1);
                }
            }
        }

        // read data of frontier pages
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto pageID = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = pageID;
                //read a whole page/sector of data
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;//num_sectors_per_node mostly is 1
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                //This is reading topo info -- offset begining with 1
                frontier_read_reqs.emplace_back((uint64_t)(pageID + 1) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
    
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }

            io_timer.reset();
#ifdef USE_BING_INFRA
            dataReader->read(frontier_read_reqs, data_ctx, true); // asynhronous reader for Bing.
#else
            dataReader->read(frontier_read_reqs, data_ctx);
#endif
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods of each cached pages
        for (auto &cached_nhood : cached_nhoods)
        {
            //cached_nhood.first is the pageID of this set of cached nhoods. if the nhood of the page is cached, the coord of the page must also have been cached
            //auto global_coord_cache_iter = _coord_cache.find(cached_nhood.first);
            //T *page_coords_copy = global_coord_cache_iter->second;//this is the vector of all nodes within the page -- nnodes_per_section * dim * T
            const uint32_t pageID = cached_nhood.first;
            const uint32_t first_node_id = static_cast<uint32_t>(pageID * _nnodes_per_sector);
            const uint32_t idx_in_buff = cached_nhood.second;
            const T *page_coords = _coord_cache_buf + idx_in_buff * coords_stride;
            const uint32_t* nhood_base = _nhood_cache_buf + idx_in_buff * nhood_stride;//nhood_stride  = _max_degree + 1;
            uint32_t nnbrs = *nhood_base;
            const uint32_t *page_nbrs = nhood_base + 1;

            uint64_t num_nodes = _nnodes_per_sector;
            //if it is the last page, it wont be _nnodes_per_sector
            if (pageID == (_num_pages - 1)){
                num_nodes = _num_points % _nnodes_per_sector;
                if (num_nodes == 0) num_nodes = _nnodes_per_sector;
            }

            //––– full precision distance pass –––
            for(uint32_t k = 0; k < num_nodes; k++){
                uint32_t nid = first_node_id + k;
                const T *cur_node_coords = page_coords + k * _data_dim;
                memcpy(data_buf, cur_node_coords, _disk_bytes_per_point);
                float cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                full_retset.push_back(Neighbor(nid, cur_expanded_dist));
            }
            full_precision_cmps += static_cast<uint32_t>(num_nodes);

            //––– collect neighbor candidates –––
            std::vector<uint32_t> candidate_nbrs_IDs;
            candidate_nbrs_IDs.reserve(nnbrs);
            for(uint32_t k = 0; k < nnbrs; k++){
                uint32_t nbrID = page_nbrs[k];
                uint32_t nbr_pid = nbrID / static_cast<uint32_t>(_nnodes_per_sector);
                if (expandedPages.find(nbr_pid) != expandedPages.end()){
                    continue;
                }

                memcpy(pq_coord_scratch + candidate_nbrs_IDs.size() * each_pq_space, _cached_pq_buff + nbrID * _n_chunks, each_pq_space);
                candidate_nbrs_IDs.push_back(nbrID);
            }

            diskann::pq_dist_lookup(pq_coord_scratch, candidate_nbrs_IDs.size(), _n_chunks, pq_dists, dist_scratch);

            for (uint64_t m = 0; m < candidate_nbrs_IDs.size(); ++m)
            {
                uint32_t nbr_id = candidate_nbrs_IDs[m];
                float dist = dist_scratch[m];
                Neighbor nn(nbr_id, dist);
                retset.insert(nn);
            }
        }

#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are
        // reads in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(data_ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*data_ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            uint64_t num_nodes = _nnodes_per_sector;
            if (frontier_nhood.first == (_num_pages - 1)){
                num_nodes = _num_points % _nnodes_per_sector;
                if (num_nodes == 0) num_nodes = _nnodes_per_sector;
            }

            uint32_t first_node_id = static_cast<uint32_t>(frontier_nhood.first * _nnodes_per_sector);
            uint32_t nbr_id = first_node_id;
            for(uint8_t k = 0; k < num_nodes; k++){
                nbr_id = first_node_id + k;
                char *node_disk_buf = frontier_nhood.second + k * _max_node_len;
                T *node_fp_coords = (T *)(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                full_retset.push_back(Neighbor(nbr_id, dist));
            }
            full_precision_cmps += static_cast<uint32_t>(num_nodes);

            // Parse neighbor information from page
            uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(frontier_nhood.second + all_full_vector_space);
            uint16_t num_uncached_nbrs = *reinterpret_cast<uint16_t*>(frontier_nhood.second + all_full_vector_space + sizeof(uint16_t));
            std::vector<uint32_t> candidate_nbrs_IDs;
            uint32_t total_num_nbrs = (uint32_t)(num_cached_nbrs) + (uint32_t)(num_uncached_nbrs);
            candidate_nbrs_IDs.reserve(total_num_nbrs);

            char *page_nbrs_IDs_buf = frontier_nhood.second + all_full_vector_space + 2 * sizeof(uint16_t);
            // Collect cached neighbor IDs
            size_t actual_num_cached_nbrs_used = 0;
            for(uint16_t k = 0; k < num_cached_nbrs && candidate_nbrs_IDs.size() < defaults::MAX_GRAPH_DEGREE; k++){
                uint32_t nbrID = *reinterpret_cast<uint32_t*>(page_nbrs_IDs_buf);
                uint32_t pageID = static_cast<uint32_t>(nbrID / _nnodes_per_sector);
                if (expandedPages.find(pageID) == expandedPages.end()){
                    candidate_nbrs_IDs.push_back(nbrID);
                    actual_num_cached_nbrs_used++;
                }
                page_nbrs_IDs_buf += sizeof(uint32_t);
            }

            // Collect uncached neighbor IDs
            std::vector<uint16_t> actual_used_uncached_nbrs_idx;
            actual_used_uncached_nbrs_idx.reserve(num_uncached_nbrs);
            for(uint16_t k = 0; k < num_uncached_nbrs && candidate_nbrs_IDs.size() < defaults::MAX_GRAPH_DEGREE; k++){
                uint32_t nbrID = *reinterpret_cast<uint32_t*>(page_nbrs_IDs_buf);
                uint32_t pageID = static_cast<uint32_t>(nbrID / _nnodes_per_sector);
                if (expandedPages.find(pageID) == expandedPages.end()){
                    candidate_nbrs_IDs.push_back(nbrID);
                    actual_used_uncached_nbrs_idx.push_back(k);
                }
                page_nbrs_IDs_buf += sizeof(uint32_t);
            }

            // Load PQ data for cached neighbors
            for(size_t i = 0; i < actual_num_cached_nbrs_used; i++){
                uint8_t *pq_coord_scratch_buf = pq_coord_scratch + i * each_pq_space;
                uint32_t nbrID = candidate_nbrs_IDs[i];
                uint32_t pd_idx = nbrID;
                if(_useID_pqIdx_map){
                    pd_idx = _nodeID_pqIdx_map[nbrID];
                }else if (_useID_pqIdx_array){
                    pd_idx = _nodeID_pqIdx_arr[nbrID];
                }
                memcpy(pq_coord_scratch_buf, _cached_pq_buff + pd_idx * _n_chunks, each_pq_space);
            }

            // Load PQ data for uncached neighbors
            if (actual_used_uncached_nbrs_idx.size() > 0){
                char *page_nbrs_pq_buf = frontier_nhood.second + all_full_vector_space + 2 * sizeof(uint16_t) + total_num_nbrs * sizeof(uint32_t);
                uint8_t *pq_coord_scratch_buf = pq_coord_scratch + actual_num_cached_nbrs_used * each_pq_space;
                for(auto idx : actual_used_uncached_nbrs_idx){
                    memcpy(pq_coord_scratch_buf, page_nbrs_pq_buf + idx * each_pq_space, each_pq_space);
                    pq_coord_scratch_buf += each_pq_space;
                }
            }

            diskann::pq_dist_lookup(pq_coord_scratch, candidate_nbrs_IDs.size(), _n_chunks, pq_dists, dist_scratch);
            for (uint64_t m = 0; m < candidate_nbrs_IDs.size(); ++m){
                uint32_t nid = candidate_nbrs_IDs[m];
                float dist = dist_scratch[m];
                Neighbor nn(nid, dist);
                retset.insert(nn);
            }

        }
        hops++;
    }

    std::sort(full_retset.begin(), full_retset.end());

    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
        }
    }

#ifdef USE_BING_INFRA
    data_ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }
}

// Linux version of page_search (blocking I/O despite original async intent)
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::linux_page_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                uint64_t *indices, float *distances, const uint64_t beam_width,
                                                const bool use_reorder_data, QueryStats *stats, const bool use_hash_routing)
{
    const uint32_t io_limit = std::numeric_limits<uint32_t>::max();

    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__, __LINE__);

    // Get scratch space from thread pool
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &data_ctx = data->data_ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    query_scratch->reset();

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
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
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
            aligned_query_T[i] = query1[i];
        }
    }
    pq_query_scratch->initialize(this->_data_dim, aligned_query_T);

    // Buffers for node data and distance calculations
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // Precompute PQ distances
    _pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;
    tsl::robin_set<uint32_t> &visitedPages = query_scratch->expandedPages;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;
    uint32_t full_precision_cmps = 0;
    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();

    bool useMediod = true;

    uint64_t each_nbr_ID_PQ_space = (uint64_t)(sizeof(uint32_t) + _n_chunks * sizeof(uint8_t));
    uint64_t each_pq_space = (uint64_t)(_n_chunks * sizeof(uint8_t));
    std::size_t coords_stride = _nnodes_per_sector * _data_dim;
    std::size_t nhood_stride  = _max_degree + 1;
    uint64_t all_full_vector_space = _nnodes_per_sector * _max_node_len;

    // LSH-based entry point selection
    if (use_hash_routing){
        useMediod = false;

        // Normalize query vector
        float norm = 0.0f;
        for (size_t j = 0; j < _data_dim; ++j) {
            norm += query_float[j] * query_float[j];
        }
        norm = std::sqrt(norm);
        float norm_inv = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
        float* normalized_query = new float[_data_dim];
        for (size_t j = 0; j < _data_dim; ++j) {
            normalized_query[j] = query_float[j] * norm_inv;
        }

        // Compute LSH hash for query
        uint32_t query_hash = 0;
        for (int i = 0; i < _numProjections; ++i) {
            float dotProduct = 0.0;
            size_t offset = i * _data_dim;
            for (size_t j = 0; j < _data_dim; ++j) {
                dotProduct += normalized_query[j] * _projectionMatrix[offset + j];
            }
            if (dotProduct >= 0) {
                query_hash |= (1U << i);
            }
        }
        delete[] normalized_query;

        // Generate neighboring hashes within Hamming radius
        tsl::robin_set<uint32_t> neighbor_hashes;
        neighbor_hashes.insert(query_hash);
        std::vector<uint32_t> temp_neighbors = {query_hash};

        for (int r = 1; r <= _radius; ++r) {
            std::vector<uint32_t> new_neighbors;
            for (uint32_t h : temp_neighbors) {
                for (int i = 0; i < _numProjections; ++i) {
                    uint32_t neighbor_hash = h ^ (1U << i);
                    if (neighbor_hashes.insert(neighbor_hash).second) {
                        new_neighbors.push_back(neighbor_hash);
                    }
                }
            }
            temp_neighbors = std::move(new_neighbors);
        }

        std::vector<uint32_t> closeNodes;
        uint8_t *pq_coord_scratch_buf = pq_coord_scratch;
        for (uint32_t hash : neighbor_hashes) {
            auto it = _buckets.find(hash);
            if (it != _buckets.end()) {
                uint32_t* buff_ptr = it->second.first;
                uint8_t num_IDs = it->second.second;
                uint32_t id;
                for (uint8_t i = 0; i < num_IDs; ++i) {
                    memcpy(&id, buff_ptr + i, sizeof(uint32_t));
                    closeNodes.push_back(id);

                    uint32_t pd_idx = id;
                    if(_useID_pqIdx_map){
                        pd_idx = _nodeID_pqIdx_map[id];
                    }else if (_useID_pqIdx_array){
                        pd_idx = _nodeID_pqIdx_arr[id];
                    }
                    memcpy(pq_coord_scratch_buf, _cached_pq_buff + pd_idx * _n_chunks, _n_chunks * sizeof(uint8_t));
                    pq_coord_scratch_buf += _n_chunks * sizeof(uint8_t);

                    if (closeNodes.size() >= defaults::MAX_GRAPH_DEGREE){
                        break;
                    }
                }
            }
            if (closeNodes.size() >= defaults::MAX_GRAPH_DEGREE){
                break;
            }
        }

        diskann::pq_dist_lookup(pq_coord_scratch, closeNodes.size(), _n_chunks, pq_dists, dist_scratch); 

        for(size_t i = 0; i < closeNodes.size(); ++i){
            uint32_t nid = closeNodes[i];
            float dist = dist_scratch[i];
            Neighbor nn(nid, dist);
            retset.insert(nn);
        }

        stats->n_lsh_entry_points += static_cast<uint32_t>(retset.size());
            
        if (retset.size() == 0) {
            useMediod = true;
        }          
    }

    // Medoid-based entry point selection
    if (useMediod){
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            uint32_t firstNodeID = static_cast<uint32_t>(_medoids[cur_m] * _nnodes_per_sector);
            for (uint8_t k = 0; k < _nnodes_per_sector; k++){
                float *node_disk_buf = _centroid_data + cur_m * _nnodes_per_sector * _data_dim + k * _data_dim;
                T *node_fp_coords = (T *)(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                uint32_t nbr_id = firstNodeID + k;
                full_retset.push_back(Neighbor(nbr_id, dist));
                if (dist < best_dist)
                {
                    best_medoid = nbr_id;
                    best_dist = dist;
                }
            }
            full_precision_cmps += (uint32_t)_nnodes_per_sector;
        }
        retset.insert(Neighbor(best_medoid, best_dist));
    }
    
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, uint32_t>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);
    std::vector<uint32_t> last_pages_ids;
    last_pages_ids.reserve(2 * beam_width);
    std::vector<char> last_pages_buffer(all_full_vector_space * beam_width * 2);
    int n_ops = 0;

    // Beam search main loop
    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;

        // Select beam
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            uint32_t pageID = static_cast<uint32_t>(nbr.id / _nnodes_per_sector);
            if (visitedPages.insert(pageID).second){
                num_seen++;
                auto iter = _cached_page_idx_map.find(pageID);
                if (iter != _cached_page_idx_map.end()){
                    cached_nhoods.emplace_back(pageID, iter->second);
                    if (stats != nullptr){
                        stats->n_cache_hits++;
                    }
                }
                else{
                    frontier.push_back(pageID);
                }

                if (this->_count_visited_pages)
                {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_page_visit_counter[pageID].second).fetch_add(1);
                }
            }
        }

        // Async read frontier pages
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto pageID = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = pageID;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back((uint64_t)(pageID + 1) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);

                if (stats != nullptr)
                {
                    stats->n_ios++;
                }
                num_ios++;
            }

            io_timer.reset();
#ifndef _WINDOWS
            n_ops = dataReader->submit_reqs(frontier_read_reqs, data_ctx);
#endif
        }

        // Process pages from previous round
        for (size_t i = 0; i < last_pages_ids.size(); ++i)
        {
            uint32_t pageID = last_pages_ids[i];
            char *page_data_buf = last_pages_buffer.data() + i * all_full_vector_space;
            uint64_t num_nodes = _nnodes_per_sector;
            if (pageID == (_num_pages - 1)){
                num_nodes = _num_points % _nnodes_per_sector;
                if (num_nodes == 0) num_nodes = _nnodes_per_sector;
            }

            uint32_t first_node_id = static_cast<uint32_t>(pageID * _nnodes_per_sector);
            uint32_t node_id = first_node_id;
            for(uint8_t k = 0; k < num_nodes; k++){
                node_id = first_node_id + k;
                char *node_disk_buf = page_data_buf + k * _max_node_len;
                T *node_fp_coords = (T *)(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                full_retset.push_back(Neighbor(node_id, dist));
            }
            full_precision_cmps += static_cast<uint32_t>(num_nodes);
        }

        last_pages_ids.clear();

        // process cached nhoods of each cached pages
        for (auto &cached_nhood : cached_nhoods)
        {
            const uint32_t pageID = cached_nhood.first;
            const uint32_t first_node_id = static_cast<uint32_t>(pageID * _nnodes_per_sector);
            const uint32_t idx_in_buff = cached_nhood.second;
            const uint32_t* nhood_base = _nhood_cache_buf + idx_in_buff * nhood_stride;//nhood_stride  = _max_degree + 1;
            uint32_t nnbrs = *nhood_base;
            const uint32_t *page_nbrs = nhood_base + 1;

            uint64_t num_nodes = _nnodes_per_sector;
            //if it is the last page, it wont be _nnodes_per_sector
            if (pageID == (_num_pages - 1)){
                num_nodes = _num_points % _nnodes_per_sector;
                if (num_nodes == 0) num_nodes = _nnodes_per_sector;
            }
            
            //––– full precision distance pass –––
            const T *page_coords = _coord_cache_buf + idx_in_buff * coords_stride;
            uint32_t nid = first_node_id;
            for(uint32_t k = 0; k < num_nodes; k++){
                nid = first_node_id + k;
                const T * cur_node_coords = page_coords + k * _data_dim;
                memcpy(data_buf, cur_node_coords, _disk_bytes_per_point);// _dist_cmp compare T while _dist_cmp_float compares float
                float cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                full_retset.push_back(Neighbor(nid, cur_expanded_dist));
            }
            full_precision_cmps += static_cast<uint32_t>(num_nodes);

            //––– collect neighbor candidates –––
            std::vector<uint32_t> candidate_nbrs_IDs;
            candidate_nbrs_IDs.reserve(nnbrs);
            for(uint32_t k = 0; k < nnbrs; k++){
                uint32_t nbrID = page_nbrs[k];
                //if not visited by retset yet, namely not add this into retset yet (check its existance through retset take logn time, which is too much)
                //if(visited.insert(nbrID).second){
                    //skip if already expaned this page. ie. not visited yet
                    uint32_t nbr_pid = nbrID / static_cast<uint32_t>(_nnodes_per_sector);
                    if (visitedPages.find(nbr_pid) != visitedPages.end()){
                        continue;
                    }
        
                    // if(_useID_pqIdx_map || _useID_pqIdx_array)//should never be true
                    memcpy(pq_coord_scratch + candidate_nbrs_IDs.size() * each_pq_space, _cached_pq_buff + nbrID * _n_chunks, each_pq_space);
                    candidate_nbrs_IDs.push_back(nbrID);
                //}
                
            }

            diskann::pq_dist_lookup(pq_coord_scratch, candidate_nbrs_IDs.size(), _n_chunks, pq_dists, dist_scratch);

            for (uint64_t m = 0; m < candidate_nbrs_IDs.size(); ++m)
            {
                uint32_t nbr_id = candidate_nbrs_IDs[m];
                float dist = dist_scratch[m];
                Neighbor nn(nbr_id, dist);
                retset.insert(nn);
            }
            //stats->nnbr_explored += static_cast<uint32_t>(candidate_nbrs_IDs.size());
        }

        // Wait for async I/O completion
#ifndef _WINDOWS
        if (!frontier.empty()) {
            dataReader->get_events(data_ctx, n_ops);
        }
#endif
        if (stats != nullptr)
        {
            stats->io_us += (float)io_timer.elapsed();
        }

        // Process frontier pages
        for (auto &frontier_nhood : frontier_nhoods)
        {
            char *sector_buf = frontier_nhood.second;
            memcpy(last_pages_buffer.data() + last_pages_ids.size() * all_full_vector_space, sector_buf, all_full_vector_space);
            last_pages_ids.push_back(frontier_nhood.first);

            uint16_t num_cached_nbrs = *reinterpret_cast<uint16_t*>(frontier_nhood.second + all_full_vector_space);
            uint16_t num_uncached_nbrs = *reinterpret_cast<uint16_t*>(frontier_nhood.second + all_full_vector_space + sizeof(uint16_t));
            std::vector<uint32_t> candidate_nbrs_IDs;
            uint32_t total_num_nbrs = (uint32_t)(num_cached_nbrs) + (uint32_t)(num_uncached_nbrs);
            candidate_nbrs_IDs.reserve(total_num_nbrs);

            char *page_nbrs_IDs_buf = frontier_nhood.second + all_full_vector_space + 2 * sizeof(uint16_t);
            size_t actual_num_cached_nbrs_used = 0;
            for(uint16_t k = 0; k < num_cached_nbrs && candidate_nbrs_IDs.size() < defaults::MAX_GRAPH_DEGREE; k++){
                uint32_t nbrID = *reinterpret_cast<uint32_t*>(page_nbrs_IDs_buf);
                uint32_t pageID = static_cast<uint32_t>(nbrID / _nnodes_per_sector);
                if (visitedPages.find(pageID) == visitedPages.end()){
                    candidate_nbrs_IDs.push_back(nbrID);
                    actual_num_cached_nbrs_used++;
                }
                page_nbrs_IDs_buf += sizeof(uint32_t);
            }

            std::vector<uint16_t> actual_used_uncached_nbrs_idx;
            actual_used_uncached_nbrs_idx.reserve(num_uncached_nbrs);
            for(uint16_t k = 0; k < num_uncached_nbrs && candidate_nbrs_IDs.size() < defaults::MAX_GRAPH_DEGREE; k++){
                uint32_t nbrID = *reinterpret_cast<uint32_t*>(page_nbrs_IDs_buf);
                uint32_t pageID = static_cast<uint32_t>(nbrID / _nnodes_per_sector);
                if (visitedPages.find(pageID) == visitedPages.end()){
                    candidate_nbrs_IDs.push_back(nbrID);
                    actual_used_uncached_nbrs_idx.push_back(k);
                }
                page_nbrs_IDs_buf += sizeof(uint32_t);
            }

            for(size_t i = 0; i < actual_num_cached_nbrs_used; i++){
                uint8_t *pq_coord_scratch_buf = pq_coord_scratch + i * each_pq_space;
                uint32_t nbrID = candidate_nbrs_IDs[i];
                uint32_t pd_idx = nbrID;
                if(_useID_pqIdx_map){
                    pd_idx = _nodeID_pqIdx_map[nbrID];
                }else if (_useID_pqIdx_array){
                    pd_idx = _nodeID_pqIdx_arr[nbrID];
                }

                memcpy(pq_coord_scratch_buf, _cached_pq_buff + pd_idx * _n_chunks, each_pq_space);
            }

            if (actual_used_uncached_nbrs_idx.size() > 0){
                char *page_nbrs_pq_buf = frontier_nhood.second + all_full_vector_space + 2 * sizeof(uint16_t) + total_num_nbrs * sizeof(uint32_t);
                uint8_t *pq_coord_scratch_buf = pq_coord_scratch + actual_num_cached_nbrs_used * each_pq_space;
                for(auto idx : actual_used_uncached_nbrs_idx){
                    memcpy(pq_coord_scratch_buf, page_nbrs_pq_buf + idx * each_pq_space, each_pq_space);
                    pq_coord_scratch_buf += each_pq_space;
                }
            }

            diskann::pq_dist_lookup(pq_coord_scratch, candidate_nbrs_IDs.size(), _n_chunks, pq_dists, dist_scratch);
            for (uint64_t m = 0; m < candidate_nbrs_IDs.size(); ++m){
                uint32_t nid = candidate_nbrs_IDs[m];
                float dist = dist_scratch[m];
                Neighbor nn(nid, dist);
                retset.insert(nn);
            }
        }
    
        hops++;
    }

    // Process final pages
    for (size_t i = 0; i < last_pages_ids.size(); ++i)
    {
        uint32_t pageID = last_pages_ids[i];
        char *page_data_buf = last_pages_buffer.data() + i * all_full_vector_space;
        uint64_t num_nodes = _nnodes_per_sector;
        if (pageID == (_num_pages - 1)){
            num_nodes = _num_points % _nnodes_per_sector;
            if (num_nodes == 0) num_nodes = _nnodes_per_sector;
        }

        uint32_t first_node_id = static_cast<uint32_t>(pageID * _nnodes_per_sector);
        uint32_t node_id = first_node_id;
        for(uint8_t k = 0; k < num_nodes; k++){
            node_id = first_node_id + k;
            char *node_disk_buf = page_data_buf + k * _max_node_len;
            T *node_fp_coords = (T *)(node_disk_buf);
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            full_retset.push_back(Neighbor(node_id, dist));
        }
        full_precision_cmps += static_cast<uint32_t>(num_nodes);
    }
    last_pages_ids.clear();

    // Sort and extract top-k results
    std::sort(full_retset.begin(), full_retset.end());

    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
        }
    }

#ifdef USE_BING_INFRA
    data_ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    //not sure if dataReader work
    IOContext &ctx = dataReader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    dataReader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
