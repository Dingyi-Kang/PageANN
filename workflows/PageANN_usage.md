# PageANN Usage Guide

Complete workflow for building and searching PageANN indexes.

PageANN organizes vectors into disk-aligned pages for efficient SSD-based retrieval. The build pipeline has four steps: build Vamana index → generate page graph → build nav graph → search.

---

## Step 1: Recommend Graph Degree Parameters

Use `apps/recommend_vamana_graph_degree` to determine optimal graph degree for a given memory budget.

| Argument | Description |
|---|---|
| `--data_path` | Input dataset in binary format |
| `--data_type` | `float`, `int8`, or `uint8` |
| `--full_ooc` | `true` for fully out-of-core processing |
| `--num_PQ_chunks` | PQ chunks for compression (typically 12–32) |
| `--mem_budget_in_GB` | Search-time memory budget in GB |
| `--min_degree_per_node` | Minimum graph degree per node |

**Example (SIFT100M):**
```bash
./build/apps/recommend_vamana_graph_degree \
  --data_path ~/sift100m/learn.100M.u8bin \
  --data_type uint8 \
  --full_ooc false \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 3.6 \
  --min_degree_per_node 23
```

---

## Step 2: Build Vamana Disk Index

Use `apps/build_vamana_disk_index` to construct the vector-level Vamana graph on disk.

| Argument | Description |
|---|---|
| `--data_type` | Data type — must match Step 1 |
| `--dist_fn` | Distance function: `l2`, `cosine`, or `mips` |
| `--data_path` | Input dataset in binary format |
| `--index_path_prefix` | Output prefix for all index files |
| `-R` | Maximum graph degree. Set equal to page-level degree per vector for best page utilization. |
| `-L` | Build-time search list size (100–200; higher = better quality, slower build) |
| `-B` | Search-time memory budget in GB. Also determines PQ chunk count: `PQ_chunks = min(dim, floor(B_bytes/N_points))` |
| `-M` | Build-time memory budget in GB |

**Example (SIFT100M):**
```bash
./build/apps/build_vamana_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --index_path_prefix /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20 \
  -R 25 \
  -L 150 \
  -B 2.2 \
  -M 40
```

---

## Step 3: Generate PageANN Page Graph

Use `apps/generate_page_graph` to convert the Vamana index into PageANN's page-aligned disk format.

| Argument | Description |
|---|---|
| `--data_type` | Data type — must match previous steps |
| `--dist_fn` | Distance function — must match Step 2 |
| `--data_path` | Dataset path |
| `--vamana_index_path_prefix` | Prefix from Step 2 output |
| `--R` | Graph degree — must match Step 2 |
| `--num_PQ_chunks` | PQ chunks — must match Step 2 |
| `--mem_budget_in_GB` | Memory budget in GB |
| `--full_ooc` | `true` for fully out-of-core processing |
| `--min_degree_per_node` | Minimum degree per vector node (from Step 1). Set equal to `-R` for full page utilization. |
| `--enable_spare_fill` | *(Optional)* Fill spare slots in each page using beam-search to find additional neighbors. Default: `true`. Set `false` to skip filling and accept pages with fewer than the maximum neighbors. |

**Example (SIFT100M):**
```bash
./build/apps/generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --vamana_index_path_prefix /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20 \
  --R 25 \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 2.2 \
  --full_ooc false \
  --min_degree_per_node 25
# --enable_spare_fill is true by default; pass --enable_spare_fill false to disable
```

Output prefix follows the pattern `<vamana_prefix>_PGD<N>_PageANN` where N = vectors per page.

---

## Step 4: Build In-Memory Nav Graph

Use `apps/utils/build_pageann_nav_graph` to build an in-memory nav graph over sampled page vectors using full-precision distances. This graph seeds the disk search with high-quality starting candidates.

| Argument | Description |
|---|---|
| `--data_type` | Data type |
| `--dist_fn` | Distance function |
| `--index_file` | Full path to the PageANN `.index` file from Step 3 |
| `--output_prefix` | Output prefix for nav graph files (typically same as index prefix) |
| `--samples_per_page` | Number of vectors to sample from **every** page (default: `1`). Sampling order: top vector (index 0) first, then last, second-to-last, etc. Guarantees one nav graph node per page at minimum. Ignored when `--num_sampled_pages` is set. |
| `--num_sampled_pages` | Sample exactly this many pages using evenly distributed selection: `selected_page[i] = i×(N−1)/(K−1)`. Takes 1 vector (top) per selected page. `0` = use `--samples_per_page` over all pages. Use when memory budget limits the total number of nav graph nodes. |
| `-R` | Nav graph max degree (typically 23–24) |
| `-L` | Build-time search list size (typically 100–150) |
| `--alpha` | Vamana pruning parameter (default: 1.2) |
| `-T` | Number of build threads |

**Example (SIFT100M, 1 sample/page — default, every page covered):**
```bash
./build/apps/utils/build_pageann_nav_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --index_file /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN.index \
  --output_prefix /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN \
  --samples_per_page 1 \
  -R 23 \
  -L 100 \
  --alpha 1.2 \
  -T 16
```

**Example (SIFT100M, sparse coverage — 5% of pages when memory is tight):**
```bash
./build/apps/utils/build_pageann_nav_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --index_file /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN.index \
  --output_prefix /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN \
  --num_sampled_pages 277778 \
  -R 23 \
  -L 100 \
  --alpha 1.2 \
  -T 16
```

Outputs: `<output_prefix>_nav_graph.index` and `<output_prefix>_nav_data.bin`.

---

## Step 5: Compute Ground Truth

Use `apps/utils/compute_groundtruth` to generate ground truth for recall evaluation. PageANN remaps vector IDs when grouping into pages, so the original Vamana GT must be remapped using the `--id_map_file` produced by `generate_page_graph`.

**Fast path — remap existing Vamana GT (no KNN recomputation):**
```bash
./build/apps/utils/compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file ~/sift100m/learn.100M.u8bin \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file /mnt/nvme/sift100m/PageANN/gt_pageann_K100.bin \
  --K 100 \
  --id_map_file /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN_new_to_old_ids_map.bin \
  --vamana_gt_file /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20_gt.bin \
  --skip_knn
```

---

## Step 6: Search

Use `apps/search_disk_index` to run queries against the PageANN index.

| Argument | Description |
|---|---|
| `--data_type` | Data type |
| `--dist_fn` | Distance function |
| `--index_path_prefix` | PageANN index prefix from Step 3 |
| `--query_file` | Query vectors in binary format |
| `--gt_file` | Ground truth file from Step 5 |
| `-K` | Number of nearest neighbors to return |
| `-L` | Search list sizes — space-separated, repeat for multiple trials |
| `-W` | Beam width: I/O requests per round (default: 5) |
| `-T` | Number of query threads |
| `--num_pages_to_cache` | Number of pages to preload into DRAM (or use `--cache_ratio`) |
| `--cache_ratio` | Fraction of dataset pages to cache in DRAM |
| `--nav_L` | Nav graph search depth — enables nav graph entry point seeding (0 = disabled) |

**Example (SIFT100M, T=16 threads):**
```bash
./build/apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file /mnt/nvme/sift100m/PageANN/gt_pageann_K100.bin \
  -K 10 \
  -L 60 60 60 60 60 \
  -W 5 \
  -T 16 \
  --cache_ratio 0.2 \
  --nav_L 10
```

---

## (Optional) Regenerate PQ with Different Compression

Change PQ chunk count without rebuilding the entire index:

```bash
./build/apps/utils/generate_reorder_pq \
  uint8 \
  ~/sift100m/learn.100M.u8bin \
  /mnt/nvme/sift100m/PageANN/vamana_sift100M_R25_L150_PQ20_PGD450_PageANN \
  16
```

Outputs `<prefix>_PQ16_pq_pivots.bin` and `<prefix>_PQ16_reorder_pq_compressed.bin`. Use `--pq_path_prefix` in search to select these files.

---

## Quick Reference: Parameter Defaults

| Parameter | Value | Notes |
|---|---|---|
| Beam width `-W` | 5 | Fixed across datasets |
| Threads `-T` | 16 | For throughput experiments |
| Nav graph depth `--nav_L` | 10 | |
| Nav graph degree `-R` | 23 | At build time |
| `--samples_per_page` | 1 | 1 vector per page (every page covered); increase for denser nav graph |
