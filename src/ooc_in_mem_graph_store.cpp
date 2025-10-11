// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// PageANN: Disk-based Index Construction and Utilities
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.

#include "ooc_in_mem_graph_store.h"
#include "utils.h"
#include "defaults.h"
#include <iomanip> 

namespace diskann
{
    //function 1
InMemOOCGraphStore::InMemOOCGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
    : AbstractGraphStore(total_pts, reserve_graph_degree)
{
    //_capacity = total_pts;
    diskann::cout << "Graph Store is initialized." << std::endl;
    _max_degree = static_cast<uint32_t>(reserve_graph_degree);
    _graph_data = new uint32_t[total_pts * reserve_graph_degree];
    _graph_nbrs_size = new uint8_t[total_pts];
}

//function 2
InMemOOCGraphStore::~InMemOOCGraphStore()
{
    if (_graph_data != nullptr)
    {
        delete[] _graph_data;
    }

    if (_graph_nbrs_size != nullptr)
    {
        delete[] _graph_nbrs_size;
    }

}

//function 3
std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load(const std::string &index_path_prefix,
                                                             const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}

int InMemOOCGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start)
{
    return 0;
}
const std::vector<location_t> &InMemOOCGraphStore::get_neighbours(const location_t lc) const
{      
    return _graph.at(lc);
}

//function 4
std::vector<location_t> InMemOOCGraphStore::get_ooc_neighbours(const location_t nodeID)
{      
    
    uint32_t* pos = _graph_data + size_t(nodeID) * _max_degree;//need to make sure the offset not overflow
    uint8_t nnbrs = *(_graph_nbrs_size + nodeID);
    return std::vector<location_t>(pos, pos + nnbrs);
}  

//function 5
uint32_t InMemOOCGraphStore::get_start(){
    return _start;
}

void InMemOOCGraphStore::increase_capacity(size_t new_capacity){
}

void InMemOOCGraphStore::add_neighbour(const location_t i, location_t neighbour_id)
{
}

void InMemOOCGraphStore::clear_neighbours(const location_t i)
{
};
void InMemOOCGraphStore::swap_neighbours(const location_t a, location_t b)
{
};

void InMemOOCGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours)
{
}

size_t InMemOOCGraphStore::resize_graph(const size_t new_size)
{
    return 0;
}

void InMemOOCGraphStore::clear_graph()
{
}

#ifdef EXEC_ENV_OLS
//wont use this function
std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load_impl(AlignedFileReader &reader, size_t expected_num_points)
{

    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;

    auto max_points = get_max_points();
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t *)header.get());
    _max_observed_degree = *((uint32_t *)(header.get() + sizeof(size_t)));
    start = *((uint32_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

    uint32_t nodes_read = 0;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}
#endif

//function 6
std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load_impl(const std::string &filename,
                                                                  size_t target_num_points)
{
    size_t file_offset = 0; // will need this for single file format support
    std::ifstream index_reader(filename, std::ios::binary);
    if (!index_reader.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    index_reader.seekg(file_offset);

    diskann::cout << "Loading vamana index graph's metadata from " << filename << "..." << std::endl;
    uint32_t nr, nc;
    uint64_t npts_64, ndims_64, medoid, max_node_len, nnodes_per_sector;
    index_reader.read((char *)&nr, sizeof(uint32_t));
    index_reader.read((char *)&nc, sizeof(uint32_t));
    index_reader.read((char *)&npts_64, sizeof(uint64_t));
    index_reader.read((char *)&ndims_64, sizeof(uint64_t));
    index_reader.read((char *)&medoid, sizeof(uint64_t));
    index_reader.read((char *)&max_node_len, sizeof(uint64_t));
    index_reader.read((char *)&nnodes_per_sector, sizeof(uint64_t));

    _start = static_cast<uint32_t>(medoid);
    _nnodes_per_sector = static_cast<uint32_t>(nnodes_per_sector);
    _node_len = ndims_64 * _type_size;

    diskann::cout << "Loading vamana index graph from " << filename << "..." << std::endl;
    diskann::cout << "nr: " << nr << std::endl;
    diskann::cout << "nc: " << nc << std::endl;
    diskann::cout << "npts_64: " << npts_64 << std::endl;
    diskann::cout << "ndims_64: " << ndims_64 << std::endl;
    diskann::cout << "medoid: " << medoid << std::endl;

    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
    uint32_t nnbrs = 0;
    uint32_t num_page_to_read = (static_cast<uint32_t>(target_num_points) + _nnodes_per_sector - 1) / _nnodes_per_sector;
    diskann::cout << "Number of sectors (pages) to read: " << num_page_to_read << std::endl;
    size_t nnodes_cached = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    uint64_t total_degree = 0;
    for (size_t i = 0; i < num_page_to_read; i++){
        if (i % 100000 == 0 || i == num_page_to_read - 1) {
            float percent = 100.0f * (i + 1) / num_page_to_read;
            diskann::cout << "\rProgress: " << std::setw(6) << std::fixed << std::setprecision(2)
                        << percent << "% (" << (i + 1) << "/" << num_page_to_read << ")"
                        << std::flush;
        }
        //read a whole page at a time // Skip metadata sector at offset 0
        file_offset = (i + 1) * defaults::SECTOR_LEN;//this wont overflow because i is size_t
        index_reader.seekg(file_offset);
        index_reader.read(reinterpret_cast<char*>(sector_buf.get()), defaults::SECTOR_LEN);
        
        //decode the data of this page
        for (size_t k = 0; k < _nnodes_per_sector; k++){
            auto node_nnbr_ptr = reinterpret_cast<uint32_t*>(sector_buf.get() + k * max_node_len + _node_len);
            nnbrs = *node_nnbr_ptr;
            total_degree += (uint64_t)nnbrs;
            max_degree = std::max(max_degree, nnbrs);
            min_degree = std::min(min_degree, nnbrs);
            if (nnbrs >= 256){
                size_t nodeID = nnodes_cached * _nnodes_per_sector + k;
                std::cerr << "Error: neighbor count (" << static_cast<int>(nnbrs)
                          << ") for node " << nodeID << " exceeds 255 (uint8_t max)." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            *(_graph_nbrs_size + nnodes_cached) = static_cast<uint8_t>(nnbrs);
            size_t offset = nnodes_cached * _max_degree;
            memcpy(reinterpret_cast<char*>(_graph_data + offset), node_nnbr_ptr + 1, nnbrs * sizeof(uint32_t));

            nnodes_cached++;
            if (nnodes_cached >= target_num_points){
                break;
            }
        }
    }
    index_reader.close();
    //diskann::cout << std::endl;
    uint32_t nnodes_cached_32 = static_cast<uint32_t>(nnodes_cached);
    diskann::cout << "\nMax degree: " << max_degree << std::endl;
    diskann::cout << "Min degree: " << min_degree << std::endl;
    diskann::cout << "Mean degree: " << static_cast<double>(total_degree) / nnodes_cached << std::endl;
    diskann::cout << "Done. Index has cached " << nnodes_cached_32 << " nodes" << std::endl;
    diskann::cout << std::endl;
    return std::make_tuple(nnodes_cached_32, _start, 0);
}

//function 7
void InMemOOCGraphStore::set_type_size(size_t size){
    _type_size = size;
}

int InMemOOCGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                                const size_t num_frozen_points, const uint32_t start)
{
    return 0;
}

size_t InMemOOCGraphStore::getGraphNodesNum() const {
    return 0;
}

size_t InMemOOCGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t InMemOOCGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

} // namespace diskann
