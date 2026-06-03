# PageANN & LAANN

Two disk-based approximate nearest neighbor search systems built for high-throughput vector search on SSDs.

## About

### PageANN ([paper](https://arxiv.org/abs/2509.25487))

**PageANN** targets two tightly coupled root causes of scalability bottlenecks in disk-based ANNS:

1. **Granularity mismatch**: prior systems traverse the graph vector-by-vector, but SSD I/O is page-granular. Each page read fetches data for many vectors, yet only the target vector's data is used — resulting in as low as 10% page utilization for DiskANN and PipeANN.
2. **Memory overhead from compressed vectors**: to avoid reading compressed neighbor values from disk during distance estimation, prior systems keep all PQ-compressed vectors in memory. This overhead scales with dataset size and becomes the binding constraint under tight memory budgets.

PageANN addresses both through a **page-node graph**: nearby vectors are grouped into 4 KB pages, and all cross-page Vamana edges from every vector in the page are merged and deduplicated into a single page-level neighbor list. Search traverses page nodes rather than individual vectors — one I/O delivers one complete graph node, achieving 100% page utilization and eliminating read amplification.

This page-level design also enables storing compressed neighbor values (PQ codes) directly on disk, which prior systems could not do. The reason prior systems were blocked: vector-level traversal requires high per-vector degree, which inflates the number of compressed values per page until they no longer fit alongside the vector data and neighbor IDs. PageANN breaks this constraint in two ways: (1) page-level traversal achieves the same search quality with fewer edges per vector, since one page read explores all co-located vectors simultaneously; (2) neighbor merging eliminates duplicate compressed values for neighbors shared by multiple vectors in the same page. Together, these free enough space within each page to co-locate PQ codes for uncached neighbors inline.

PageANN then applies a **budget-aware placement strategy**: compressed values for frequently-referenced neighbors are cached in memory (saving page space for each page that references them); all others are stored inline on disk and retrieved for free as part of the page read — zero extra I/O. An in-memory Vamana graph built over sampled page centroids seeds the disk search with high-quality entry points before the first I/O is issued.

See the Performance section below for benchmark results.

### LAANN ([paper](https://arxiv.org/abs/2606.02784))

**LAANN** extends PageANN with three complementary techniques that make graph search explicitly I/O-aware:

- **Look-ahead search**: Adapts the search strategy across query phases — prioritizing in-memory neighbors in the approach phase to reduce wasted I/Os, and maximizing I/O parallelism in the convergence phase to minimize tail latency.
- **Priority I/O–CPU pipeline**: Fills CPU idle time during I/O waits with deferred work ordered by relevance to the next I/O decision, supported by a customized candidate pool with an overflow area.
- **Lightweight in-memory graph index**: A Vamana graph over page centroids, searched using compressed vectors, that seeds the disk graph candidate pool with high-quality candidates before the first I/O is issued.

### Disk Index Format

Both systems group vectors into 4 KB pages and build a **merged page-level neighbor list** — all cross-page Vamana edges from every vector in the page are unioned and deduplicated into a single list. One I/O read delivers one complete graph node. The two systems differ in what is stored alongside those neighbor IDs on disk:

**PageANN** stores PQ-compressed values for uncached neighbors inline within the page:

```
[ vectors | num_cached (u16) | num_uncached (u16) | cached IDs | uncached IDs | uncached PQ data ]
```

Neighbor IDs are split into two groups: those whose PQ codes are cached in memory (ID only on disk), and those whose PQ codes live on disk (ID + PQ data, both within the same page). Two counters at the start of the neighbor region record the size of each group. During search, uncached neighbors' PQ codes are read from the page itself — no extra I/O. The budget-aware placement strategy (described above) determines which neighbors fall into each group based on reference frequency across all pages.

This inline PQ design is only feasible because page-level traversal needs fewer edges per vector and neighbor merging removes duplicates — together freeing enough page space for the PQ data. Prior vector-level systems such as DiskANN cannot store PQ codes on disk: with per-vector degrees of 23–31, the combined PQ data for all a vector's neighbors exceeds the remaining page space after storing the vector and neighbor IDs.

**LAANN** stores only neighbor IDs — no PQ data on disk:

```
[ vectors | num_neighbors (u32) | neighbor IDs ]
```

PQ lookups always go through the separately loaded in-memory compressed vector file. Dropping inline PQ entirely frees the spare page space for more neighbor IDs, removing the coupling between memory budget and page capacity.

Because the binary sector layout differs, PageANN and LAANN produce separate index files and cannot share a disk index.

---

### Key Differences from Prior Disk-Based ANNS Systems

| Property | DiskANN | Starling / MARGO | PipeANN | PageANN | **LAANN** |
|---|---|---|---|---|---|
| Page-aligned disk graph | | | | ✓ | ✓ |
| Cached graph nodes support | ✓ | ✓ | | ✓ | ✓ |
| Phase-aware search strategy | | | ✓ | | ✓ |
| Relevance-ordered CPU pipeline | | | | | ✓ |
| In-memory graph (full-precision dist.) | | ✓ | ✓ | ✓ | |
| Lightweight in-memory graph (approx. dist.) | | | | | ✓ |

### Performance

#### PageANN vs. SOTA Disk-Based ANNS

Evaluated against DiskANN, Starling, MARGO, PipeANN, and SPANN across five datasets (up to 1B vectors) and memory budgets ranging from near-zero (0.004%) to 40% of dataset size:

| Metric | Peak | Average |
|---|---|---|
| Throughput improvement | 5.4× | 1.94× |
| Latency reduction | 83% | 66.8% |

Results hold at equal recall without sacrificing accuracy.

#### LAANN vs. PageANN and DiskANN

At Recall@10 = 0.9 on 100M-scale datasets (memory budget = 0.5× dataset size, T=16 threads):

| Dataset | vs. PageANN (QPS) | vs. DiskANN (QPS) | I/O reduction vs. PageANN |
|---|---|---|---|
| SIFT100M | 1.41× | 3.47× | 37% fewer |
| SPACEV100M | 2.12× | 4.66× | 58% fewer |
| DEEP100M | 1.96× | 3.36× | 58% fewer |

On billion-scale datasets the advantage widens further due to longer search paths and higher I/O contention.

## Quick Demo: SIFT1M

Try either system on the SIFT1M benchmark (1M × 128-dim float vectors).

**Download the dataset** (see [sift1M/README.md](sift1M/README.md)):
```bash
cd sift1M
wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs
wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs
wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs
```

### PageANN Demo

**Build** — data conversion, Vamana graph, PageANN page graph, and nav graph:
```bash
bash sift1M/build_pageann_sift1m.sh
```

**Search** — sweeps multiple L values to produce a recall–latency curve:
```bash
bash sift1M/search_pageann_sift1m.sh
```

See **[sift1M/build_pageann_sift1m.sh](sift1M/build_pageann_sift1m.sh)** and **[sift1M/search_pageann_sift1m.sh](sift1M/search_pageann_sift1m.sh)** for full parameter details.

### LAANN Demo

**Build** — data conversion, Vamana graph, LAANN page graph, nav graph, and page reordering:
```bash
bash sift1M/build_laann_sift1m.sh
```

**Search** — sweeps multiple L values to produce a recall–latency curve:
```bash
bash sift1M/search_laann_sift1m.sh
```

See **[sift1M/build_laann_sift1m.sh](sift1M/build_laann_sift1m.sh)** and **[sift1M/search_laann_sift1m.sh](sift1M/search_laann_sift1m.sh)** for full parameter details.

## Papers

If you use this code, please cite the relevant paper(s):

**PageANN:**
```bibtex
@article{kang2025scalable,
  title={Scalable Disk-Based Approximate Nearest Neighbor Search with Page-Aligned Graph},
  author={Kang, Dingyi and Jiang, Dongming and Yang, Hanshen and Liu, Hang and Li, Bingzhe},
  journal={arXiv preprint arXiv:2509.25487},
  year={2025}
}
```

**LAANN:**
```bibtex
@article{kang2026laann,
  title={LAANN: I/O-Aware Look-Ahead Search for Disk-Based Approximate Nearest Neighbor Search},
  author={Kang, Dingyi and Yang, Juncheng and Li, Bingzhe},
  journal={arXiv preprint arXiv:2606.02784},
  year={2026}
}
```

## Credits

Both PageANN and LAANN are built on top of Microsoft Research's DiskANN:
- DiskANN repository: https://github.com/microsoft/DiskANN
- Original DiskANN paper: Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019

**Original DiskANN**: Copyright (c) Microsoft Corporation  
**PageANN / LAANN modifications**: Copyright (c) 2025 Dingyi Kang

## Authors

**PageANN** ([arXiv:2509.25487](https://arxiv.org/abs/2509.25487)):  
Dingyi Kang, Dongming Jiang, Hanshen Yang, Hang Liu, Bingzhe Li — University of Texas at Dallas

**LAANN** ([arXiv:2606.02784](https://arxiv.org/abs/2606.02784)):  
Dingyi Kang, Juncheng Yang (Harvard University), Bingzhe Li — University of Texas at Dallas

Contact: dingyikangosu@gmail.com

## License

MIT License — see [LICENSE](LICENSE) file for details.

This project contains:
- DiskANN components: Copyright (c) Microsoft Corporation
- PageANN / LAANN modifications: Copyright (c) 2025 Dingyi Kang

## Building

### Prerequisites

**Linux (Ubuntu 20.04+)**:
```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
sudo apt install libmkl-full-dev
```

**Earlier Ubuntu versions**: Install Intel MKL manually from the [oneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).

### Build Instructions

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Usage

### PageANN

- **[PageANN Usage Guide](workflows/PageANN_usage.md)**: Complete workflow for building and searching PageANN indexes
- **[PageANN Paper Experiment Parameters](workflows/PageANN_paper_params.md)**: Exact parameters to reproduce PageANN paper results

### LAANN

- **[LAANN Usage Guide](workflows/LAANN_usage.md)**: Complete workflow for building and searching LAANN indexes
- **[LAANN Paper Experiment Parameters](workflows/LAANN_paper_params.md)**: Exact parameters to reproduce LAANN paper results

## Key Tools

### Main Applications

- `build_vamana_disk_index`: Build Vamana vector-level disk index (first step)
- `generate_laann_graph`: Convert Vamana index to LAANN page-graph format with nav-graph construction
- `generate_page_graph`: Convert Vamana index to PageANN page-graph format (baseline)
- `recommend_vamana_graph_degree`: Recommend optimal graph degree parameters for a given memory budget
- `search_disk_index`: Search the LAANN or PageANN index

### Utility Tools

- `build_pageann_nav_graph`: Build the in-memory navigation graph for PageANN
- `build_laann_nav_graph`: Build the lightweight in-memory navigation graph for LAANN
- `compute_groundtruth`: Compute ground truth nearest neighbors for recall evaluation
- `generate_reorder_pq`: Regenerate PQ codes with different compression levels
- `reorder_pages_by_frequency`: Reorder disk pages by access frequency for faster cache warm-up

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.

## Acknowledgments

Special thanks to Microsoft Research for open-sourcing DiskANN, which forms the foundation of this work.
