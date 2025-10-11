// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "abstract_graph_store.h"
#include "tsl/robin_map.h"
#include "aligned_file_reader.h"

namespace diskann
{

class InMemOOCGraphStore : public AbstractGraphStore
{
  public:
    InMemOOCGraphStore(const size_t total_pts, const size_t reserve_graph_degree);
    ~InMemOOCGraphStore();
    // returns tuple of <nodes_read, start, num_frozen_points>
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                        const size_t num_points) override;
    virtual int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                      const uint32_t start) override;

    virtual const std::vector<location_t> &get_neighbours(const location_t i) const override;
    virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
    virtual void clear_neighbours(const location_t i) override;
    virtual void swap_neighbours(const location_t a, location_t b) override;

    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) override;

    virtual size_t resize_graph(const size_t new_size) override;
    virtual void clear_graph() override;

    virtual size_t get_max_range_of_graph() override;
    virtual uint32_t get_max_observed_degree() override;
    //virtual uint32_t getMaxDegree(const size_t num_points) const override;
    virtual size_t getGraphNodesNum() const override;

    void set_type_size(size_t size);

    uint32_t get_start();

    std::vector<location_t> get_ooc_neighbours(const location_t lc);

    void increase_capacity(size_t new_capacity);
    //size_t _cache_miss = 0;

  protected:
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(AlignedFileReader &reader, size_t expected_num_points);
#endif

    int save_graph(const std::string &index_path_prefix, const size_t active_points, const size_t num_frozen_points,
                   const uint32_t start);
  private:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;
    std::vector<std::vector<uint32_t>> _graph;

    uint32_t* _graph_data = nullptr;
    uint8_t* _graph_nbrs_size = nullptr;
    uint32_t _max_degree = 0;
    uint32_t _start = 0;
    //uint32_t _max_node_len = 0;
    uint32_t _nnodes_per_sector = 0;
    size_t _type_size = 0;
    uint64_t _node_len = 0;
};

} // namespace diskann
