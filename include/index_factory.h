// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// PageANN: Page-level Graph Index Generation
// Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
// Licensed under the MIT license.
#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"
#include "in_mem_data_store.h"
#include "ooc_in_mem_graph_store.h"
#include "ooc_in_mem_data_store.h"
#include "pq_data_store.h"

namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT explicit IndexFactory(const IndexConfig &config);
    DISKANN_DLLEXPORT std::unique_ptr<AbstractIndex> create_instance();

    DISKANN_DLLEXPORT static std::unique_ptr<AbstractGraphStore> construct_graphstore(
        const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

    DISKANN_DLLEXPORT static std::shared_ptr<InMemOOCGraphStore> construct_ooc_graphstore(
        const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy,
                                                                                       size_t num_points,
                                                                                       size_t dimension, Metric m);
    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<InMemOOCDataStore<T>> construct_ooc_datastore(DataStoreStrategy stratagy,
                                                                                       size_t num_points,
                                                                                       size_t dimension, Metric m);
    // For now PQDataStore incorporates within itself all variants of quantization that we support. In the
    // future it may be necessary to introduce an AbstractPQDataStore class to spearate various quantization
    // flavours.
    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<PQDataStore<T>> construct_pq_datastore(DataStoreStrategy strategy,
                                                                                    size_t num_points, size_t dimension,
                                                                                    Metric m, size_t num_pq_chunks,
                                                                                    bool use_opq);
    template <typename T> static Distance<T> *construct_inmem_distance_fn(Metric m);

  private:
    void check_config();

    template <typename data_type, typename tag_type, typename label_type>
    std::unique_ptr<AbstractIndex> create_instance();

    std::unique_ptr<AbstractIndex> create_instance(const std::string &data_type, const std::string &tag_type,
                                                   const std::string &label_type);

    template <typename data_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &tag_type, const std::string &label_type);

    template <typename data_type, typename tag_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &label_type);

    std::unique_ptr<IndexConfig> _config;
};

} // namespace diskann
