# PageANN

A page-level approximate nearest neighbor search system optimized for SSD-based workloads.

## About

PageANN extends Microsoft's [DiskANN](https://github.com/microsoft/DiskANN) with page-level graph organization to reduce random I/O and improve SSD utilization during vector search.

### Key Differences from DiskANN

- **Page-level indexing**: Multiple vectors are merged into disk-aligned pages
- **Reduced I/O**: Fewer random accesses during graph traversal
- **SSD-optimized**: Better utilization of SSD bandwidth and parallelism

## Citation

If you use PageANN in your research, please cite:

```bibtex
@article{kang2025pageann,
  title={Scalable Disk-Based Approximate Nearest Neighbor Search with Page-Aligned Graph},
  author={Kang, Dingyi and Jiang, Dongming and Yang, Hanshen and Liu, Hang and Li, Bingzhe},
  journal={arXiv preprint arXiv:2509.25487},
  year={2025},
  url={https://www.arxiv.org/abs/2509.25487}
}
```

## Credits

This project is built upon Microsoft Research's DiskANN:
- DiskANN repository: https://github.com/microsoft/DiskANN
- Original paper: Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019

**Original DiskANN**: Copyright (c) Microsoft Corporation
**PageANN Modifications**: Copyright (c) 2025 Dingyi Kang

## Author

**Dingyi Kang**
Email: dingyikangosu@gmail.com

## License

MIT License - see [LICENSE](LICENSE) file for details.

This project contains:
- DiskANN components: Copyright (c) Microsoft Corporation
- PageANN modifications: Copyright (c) 2025 Dingyi Kang

## Building PageANN

### Prerequisites

**Linux (Ubuntu 20.04+)**:
```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
sudo apt install libmkl-full-dev
```

**Earlier Ubuntu versions**: Install Intel MKL manually from [oneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

### Build Instructions

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Usage

For detailed workflow and command-line instructions, see:
- **[PageANN Usage Guide](workflows/PageANN_usage.md)**: Complete workflow for building and searching PageANN indexes
- **[Paper Experiments Parameters](workflows/Paper%20experiments%20parameters.md)**: Exact parameters to reproduce results from our paper

### 1. Build Vamana Vector-Level Index

```bash
./build_vamana_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path base_vectors.bin \
  --index_path_prefix index_vamana \
  -R 64 -L 100 \
  -B 8.0 -M 16.0
```

### 2. Generate Page-Level Graph

```bash
./generate_page_graph \
  --data_type float \
  --dist_fn l2 \
  --data_path base_vectors.bin \
  --vamana_index_path_prefix index_vamana \
  --min_degree_per_node 32 \
  --num_PQ_chunks 12 \
  --R 64 \
  --mem_budget_in_GB 8.0 \
  --full_ooc false
```

### 3. Search Page Index

```bash
./search_disk_index \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix page_index \
  --query_file queries.bin \
  --gt_file ground_truth.bin \
  -K 10 -L 50 100 150 \
  --result_path results.txt
```

## Key Tools

- `build_vamana_disk_index`: Build Vamana vector-level disk index
- `generate_page_graph`: Convert Vamana index to page-level graph
- `recommend_vamana_graph_degree`: Recommend optimal graph degree parameters
- `search_disk_index`: Search the page-level index

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

Special thanks to Microsoft Research for open-sourcing DiskANN, which forms the foundation of this work.
