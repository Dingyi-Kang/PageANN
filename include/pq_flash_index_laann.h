// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "linux_async_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <atomic>

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

template <typename T, typename LabelT = uint32_t> class PQFlashIndexLAANN
{
  public:
    DISKANN_DLLEXPORT PQFlashIndexLAANN(std::shared_ptr<AlignedFileReader> &dataReader, diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndexLAANN();

    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix, const std::string &pq_path_prefix, uint32_t beam_width);
    
    // Load first num_pages_to_cache pages sequentially into memory cache
    // Since pages are reordered by frequency, this caches the most frequently visited pages
    DISKANN_DLLEXPORT void load_sequential_page_cache(uint32_t num_pages_to_cache);

    // Load pages by order file (non-contiguous disk reads) into memory cache
    // For indexes NOT reordered by frequency; reads each page from its original disk position
    // Order file format: [num_pages uint32][page_id uint32, freq uint32]... (same as profile output)
    DISKANN_DLLEXPORT void load_order_file_page_cache(const std::string &order_file, uint32_t num_pages_to_cache);

    DISKANN_DLLEXPORT void load_nav_neighbors(const char *index_prefix);


    // Profile page visit frequency using sample queries and write sorted results to file
    DISKANN_DLLEXPORT void profile_page_frequency(std::string sample_bin, uint64_t l_search,
                                                   uint64_t beam_width, uint32_t nthreads,
                                                   const std::string &output_file, const uint32_t nav_L = 0);


    DISKANN_DLLEXPORT void laann_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                            uint64_t *res_ids, float *res_dists,
                                            const uint64_t search_beam_width,
                                            const bool use_look_ahead_search, const bool use_pipeline = true,
                                            const uint32_t persistence_window_width = 0,
                                            const uint32_t nav_L = 14, const bool use_reorder_data = false,
                                            const float retset_capacity_ratio = 2.0f,
                                            const uint32_t stable_rank_threshold = 10,
                                            const float beamwidth_spike_ratio = 0.5f,
                                            const float beam_decay_ratio = 0.9f,
                                            QueryStats *stats = nullptr);

    std::shared_ptr<AlignedFileReader> &dataReader;

    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<uint32_t *> &nbr_buffers);

    DISKANN_DLLEXPORT std::vector<bool> read_mega_node(const std::vector<uint32_t> &mega_nodes_ids, std::vector<T *> &coord_buffers, std::vector<uint32_t *> &nbr_buffers);

    DISKANN_DLLEXPORT uint64_t get_num_points();
    DISKANN_DLLEXPORT uint64_t get_num_mega_nodes();
    DISKANN_DLLEXPORT uint64_t get_num_cached_mega_nodes();

  protected:
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t estimated_visited_vectors, uint64_t beam_width);

  private:
    //is called internally by the load() function 
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath, uint32_t beam_width);
    // returns region of `node_buf` containing [COORD(T)]
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    uint64_t _max_node_len = 0;
    uint64_t _each_vector_nbrs_space = 0;
    uint64_t _num_vectors_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    uint64_t _max_vector_degree = 0; 
    uint64_t _megaNode_capacity = 0;
    uint64_t _max_page_degree = 0;

    uint64_t _each_pq_space = 0;
    size_t _pageNode_vectors_size_in_T = 0;
    uint64_t _all_vectors_space_per_pageNode = 0;
    uint64_t _pageNode_nbrs_size_in_u32 = 0;
    uint64_t _all_nbrs_space_per_pageNode = 0;
    size_t _each_vector_nbrs_offset = 0;
    uint64_t _num_sectors_per_node = 0;

    uint8_t* _cached_pq_buff = nullptr;

    ///MARK: Navigation graph data structures (sampled vectors with PQ distance)
    // Nav graph stores neighbors as original vector IDs for unified search
    uint32_t* _nav_nbrs = nullptr;             // Flat adjacency list (original vector IDs), indexed by nav_idx * _nav_uniform_degree
    uint32_t _nav_num_points = 0;              // Number of sampled vectors (total nav nodes)
    uint32_t _nav_uniform_degree = 0;          // Uniform degree for all nodes (stride for _nav_nbrs)
    uint32_t _nav_medoid_original_id = 0;      // Entry point (original vector ID)
    uint32_t _nav_samples_per_page = 0;        // Number of samples per page
    std::vector<uint32_t> _nav_most_page_sample_indices;  // Sample indices for most pages [0, cap-1, cap-2, ...]
    std::vector<uint32_t> _nav_last_page_sample_indices;  // Sample indices for last page (may have fewer vectors)
    uint32_t _nav_total_pages = 0;             // Total pages in index; >0 means page subsampling mode
    std::vector<uint32_t> _nav_selected_pages; // Selected page IDs in page subsampling mode
    bool _nav_graph_loaded = false;            // Whether nav graph is loaded

    ///MARK: Page cache data structures (for disk page caching)
    char* _cached_page_data = nullptr;         // Cached full page data (vectors + merged neighbors)
    uint32_t _num_cached_pages = 0;            // Number of pages cached

    // Non-contiguous cache (order-file mode): maps page_id -> slot index in _cached_page_data buffer
    std::unordered_map<uint32_t, uint32_t> _cached_page_id_to_slot;
    bool _use_order_file_cache = false;        // true = map-based lookup; false = sequential (page_id < _num_cached_pages)

    inline bool is_page_cached(uint32_t pageID) const {
        if (_use_order_file_cache)
            return _cached_page_id_to_slot.count(pageID) > 0;
        return pageID < _num_cached_pages;
    }

    // Returns pointer to the cached page data for pageID. Caller must ensure is_page_cached(pageID) == true.
    inline char* get_cached_page_ptr(uint32_t pageID) const {
        if (_use_order_file_cache) {
            auto it = _cached_page_id_to_slot.find(pageID);
            return _cached_page_data + static_cast<size_t>(it->second) * defaults::SECTOR_LEN;
        }
        return _cached_page_data + static_cast<size_t>(pageID) * defaults::SECTOR_LEN;
    }

    diskann::Metric metric = diskann::Metric::L2;

    // data info
    uint64_t _num_points = 0;
    uint64_t _num_mega_nodes = 0;
    uint64_t _last_mega_node_index = 0;
    uint64_t _last_page_vector_count = 0;  // Precomputed number of internal vectors in last page
    uint64_t _data_dim = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _bytes_per_vector = 0; // Number of bytes

    std::string _disk_index_file;
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;
    std::vector<std::pair<uint32_t, uint32_t>> _mega_node_visit_counter;
    tsl::robin_set<uint32_t> _top_hops_pages;

    // _n_chunks = # of chunks ndims is split into
    // so, in total, there are 256 (2^8) * numChuck centroids. and each centroid correspond to a float
    // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
    uint64_t _n_chunks; //this is the number of PQ chunk -- size of compressed pq data
    FixedChunkPQTable _pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;


    // medoid/start info
    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *_medoids = nullptr; //has manual delete
    // defaults to 1
    size_t _num_medoids;

    // thread-specific scratch
    ConcurrentQueue<SSDThreadData<T> *> _thread_data; //has manual delete
    uint64_t _max_nthreads;
    bool _load_flag = false;
    //bool _count_visited_nodes = false;
    bool _count_visited_megaNodes = false;
    bool _getMostFrequentlyVisitedNodes = false;

#ifdef __linux__
    // Async I/O Configuration (io_uring)
    // ============================================
    bool _use_async_io = false;  // Enable async I/O with batching

#endif // __linux__

};
} // namespace diskann
