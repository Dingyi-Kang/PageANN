# LAANN Usage Guide

Complete workflow for building and searching LAANN indexes.

LAANN's build pipeline has four steps: build Vamana index → generate LAANN page graph → build nav graph → (optional) reorder pages by frequency. Search follows.

---

## Step 1: Build Vamana Disk Index

Use `apps/build_vamana_disk_index` to construct the vector-level Vamana graph on disk.

| Argument | Description |
|---|---|
| `--data_type` | `float`, `int8`, or `uint8` |
| `--dist_fn` | Distance function: `l2`, `cosine`, or `mips` |
| `--data_path` | Input dataset in binary format |
| `--index_path_prefix` | Output prefix for all index files |
| `-R` | Maximum degree per vector in the Vamana graph (max neighbors). Also controls PQ chunk count via its interaction with `-B` — set R = page-level degree per vector (see Step 2). |
| `-L` | Search list size during graph construction. A larger candidate pool during greedy search produces higher-quality edges at the cost of a slower build. Typically 100–200; must be ≥ R. |
| `-B` | Search-time DRAM budget in GB. Controls how many graph nodes can be cached in memory during search, and **determines the PQ chunk count**: `PQ_chunks = min(dim, floor(B_bytes / N_points))`. The `--num_PQ_chunks` in Step 2 must match this derived value. |
| `-M` | Build-time DRAM budget in GB. Controls how much of the dataset is loaded into memory during graph construction. Larger M → faster build. |

**Example (SIFT100M, 0.5× memory budget ≈ 5.95 GB):**
```bash
./build/apps/build_vamana_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --index_path_prefix /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20 \
  -R 25 \
  -L 150 \
  -B 5.95 \
  -M 90
```

> **`-B` determines PQ chunk count**: `PQ_chunks = min(dim, floor(B_bytes / N_points))`. For SIFT100M (N=100M, dim=128, B=5.95 GB): `min(128, floor(6.39×10⁹/10⁸)) = min(128, 63) = 63`. The `--num_PQ_chunks` in Step 2 **must match** this derived value — passing a different number causes a build error. To target a specific PQ size, set `B = target_chunks × N_points / 1024³`.

> To find the recommended R and PQ chunks for a given memory budget, use `apps/recommend_vamana_graph_degree`.

---

## Step 2: Generate LAANN Page Graph

Use `apps/generate_laann_graph` to convert the Vamana index into LAANN's page-aligned disk graph. Output files land alongside the Vamana prefix and should be moved to a separate LAANN directory afterward.

| Argument | Description |
|---|---|
| `--data_type` | Data type — must match Step 1 |
| `--dist_fn` | Distance function — must match Step 1 |
| `--data_path` | Dataset path |
| `--vamana_index_path_prefix` | Prefix from Step 1 output |
| `--R` | Graph degree — must match Step 1 |
| `--num_PQ_chunks` | Number of PQ bytes per vector (1 byte per chunk). **Must equal the value derived by Step 1 from `-B`**: `min(dim, floor(B_bytes / N_points))`. Mismatch causes a build error. |
| `--L` (grouping L) | Search list size used during the page-grouping phase to find spatially close neighbors for each page. Must be ≥ vectors_per_page. Higher → tighter spatial grouping, slower build. Distinct from the Vamana build's `-L`. |
| `--fill_L` | Search list size used during the neighbor-filling phase after grouping. Higher → better page-graph edge quality, slower build. Default: 100. |
| `--min_degree_per_node` | Minimum average Vamana edges per vector the page graph must satisfy. **Set equal to `-R` from Step 1.** This ensures page capacity exactly matches the Vamana graph quality — Vamana provides R edges per vector which perfectly fills the page (`page_degree = R × vectors_per_page`). If set lower than R, the page fits more vectors but spare neighbor slots must be filled with lower-quality edges. Appears in the output filename as `MGD<min_degree × vectors_per_page>`. |

**Example (SIFT100M):**
```bash
./build/apps/generate_laann_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --vamana_index_path_prefix /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20 \
  --R 25 \
  --num_PQ_chunks 20 \
  --L 350 \
  --fill_L 200 \
  --min_degree_per_node 25
```

**Choosing R**: set Vamana `-R` = page-level degree per vector = `page_degree / vectors_per_page`. With this, the Vamana graph already provides all page edges at full quality and the fill phase only does minor cleanup. If R < page_degree/vectors_per_page, the fill phase must add low-quality edges to make up the difference.

The output prefix follows the pattern `<vamana_prefix>_MGD<N>_LAANN` where `N = min_degree_per_node × vectors_per_page`. The number of vectors per page is determined automatically from the page size (4 KB), vector size, PQ size, and `min_degree_per_node`. Typical values:

| Dataset | Data type | Dims | PQ chunks | Vectors/page |
|---|---|---|---|---|
| SIFT100M | uint8 | 128 | 20 | 18 |
| SPACEV100M | int8 | 100 | 18 | 20 |
| DEEP100M | float | 96 | 60 | 8 |
| SIFT1M | float | 128 | 26 | 6 |

Move all `_MGD*_LAANN*` files to your LAANN output directory before proceeding.

---

## Step 3: Build Lightweight In-Memory Nav Graph

Use `apps/utils/build_laann_nav_graph` to build the lightweight nav graph over page centroids. This graph seeds the disk search with high-quality candidates before the first I/O is issued.

| Argument | Description |
|---|---|
| `--data_type` | Data type |
| `--dist_fn` | Distance function |
| `--laann_disk_index_file` | Full path to the `.index` file from Step 2 |
| `--samples_per_page` | Number of vectors sampled per page (default: 1, the top vector). Higher values increase nav graph quality but cost more memory. Ignored when `--num_sampled_pages` is set. |
| `--num_sampled_pages` | Sample only this many pages out of all pages, distributed evenly using `selected_page[i] = i × (N−1) / (K−1)`. Takes the top vector from each selected page. Use when memory is insufficient to sample one vector per page (e.g. `--num_sampled_pages 100000` for 60% of SIFT1M's 166,667 pages). Default: 0 (disabled — all pages are sampled). |
| `-R` | Maximum degree of the nav graph (max neighbors per centroid node). Independent of the Vamana R — typically 24. |
| `-L` | Search list size during nav graph construction. Higher → better nav graph quality, slower build. Typically 150. |
| `--alpha` | Vamana pruning parameter for nav graph edge selection. Controls how aggressively long-range edges are pruned: α=1.0 keeps only the shortest edges; α>1.0 retains longer edges that improve graph connectivity. Paper default: 1.2. |
| `-T` | Number of build threads. |

**Example (SIFT100M):**
```bash
./build/apps/utils/build_laann_nav_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --laann_disk_index_file /mnt/nvme/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN.index \
  --samples_per_page 1 \
  -R 24 \
  -L 150 \
  --alpha 1.2 \
  -T 16
```

---

## Step 4: Ground Truth and Page Reordering

`generate_laann_graph` reassigns vector IDs when grouping vectors into pages, making the original Vamana ground truth invalid for the LAANN index. You must produce a valid GT before searching. There are two paths:

### Path A — With page reordering (recommended)

Use `apps/utils/reorder_pages_by_frequency` to physically reorder disk pages by visit frequency. This eliminates the residency hashtable overhead, speeds up cache warm-up, and remaps the GT in one step.

| | Value |
|---|---|
| Index prefix for search | `<prefix>_fsort` |
| Ground truth for search | `<prefix>_fsort_gt.bin` (produced by this tool) |

| Argument | Description |
|---|---|
| `--data_type` | Data type |
| `--dist_fn` | Distance function |
| `--index_path_prefix` | LAANN index prefix from Step 2 (without `_fsort`) |
| `--data_bin` | Dataset path |
| `--sample_ratio` | Fraction of dataset used to profile page visit frequency (default: 0.01) |
| `--orig_pq_compressed_file` | PQ compressed vectors from the Vamana build |
| `--orig_gt_file` | Original ground truth from Step 1 — will be remapped |

```bash
./build/apps/utils/reorder_pages_by_frequency \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix /mnt/nvme/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN \
  --data_bin ~/sift100m/learn.100M.u8bin \
  --sample_ratio 0.01 \
  --orig_pq_compressed_file /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20_pq_compressed.bin \
  --orig_gt_file /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20_gt.bin
```

### Path B — Without page reordering

If you skip `reorder_pages_by_frequency`, use `apps/utils/compute_groundtruth` with `--skip_knn` to remap the existing Vamana GT IDs to the page-grouped ordering using the `_new_to_old_ids_map.bin` produced by `generate_laann_graph`. This is fast — no KNN recomputation.

| | Value |
|---|---|
| Index prefix for search | `<prefix>` (no `_fsort`) |
| Ground truth for search | output `--gt_file` from the command below |

| Argument | Description |
|---|---|
| `--base_file` | Original dataset binary file |
| `--query_file` | Query vectors binary file |
| `--gt_file` | Output path for the remapped ground truth |
| `--K` | Number of ground truth neighbors per query |
| `--id_map_file` | `_new_to_old_ids_map.bin` from `generate_laann_graph` — maps original vector IDs to page-grouped IDs |
| `--vamana_gt_file` | Original Vamana ground truth to remap |
| `--skip_knn` | Skip KNN recomputation and only remap IDs — requires `--vamana_gt_file` and `--id_map_file` |

```bash
./build/apps/utils/compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file ~/sift100m/learn.100M.u8bin \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file /mnt/nvme/sift100m/LAANN/laann_gt.bin \
  --K 100 \
  --id_map_file /mnt/nvme/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_new_to_old_ids_map.bin \
  --vamana_gt_file /mnt/nvme/sift100m/diskann/vamana_sift100M_R25_L150_PQ20_gt.bin \
  --skip_knn
```

---

## Step 5: Search

Use `apps/search_disk_index` to run queries. The index prefix and ground truth file depend on whether Step 4 was run (see table above).

| Argument | Description |
|---|---|
| `--data_type` | Data type |
| `--dist_fn` | Distance function |
| `--index_path_prefix` | LAANN index prefix — use `<prefix>_fsort` if Step 4 (Path A) was run, otherwise `<prefix>` |
| `--query_file` | Query vectors in binary format |
| `--gt_file` | Ground truth file — use `<prefix>_fsort_gt.bin` (Path A) or the remapped GT from `compute_groundtruth` (Path B) |
| `-K` | Number of nearest neighbors to retrieve per query |
| `-L` | Search list size(s) — space-separated integers; each value runs a separate trial. Pass multiple values to sweep the recall–latency curve (e.g. `40 60 80 100`). |
| `-W` | Beam width: number of I/O requests issued per search round. Higher W → more I/O parallelism but more total I/Os. Paper default: 5. |
| `-T` | Number of concurrent query threads. |
| `--use_laann` | Enable LAANN search mode: activates the look-ahead search and priority I/O–CPU pipeline. Without this flag, standard greedy beam search is used (PageANN mode). |
| `--cache_ratio` | Fraction of dataset pages to preload into DRAM, sorted by visit frequency. E.g. 0.172 ≈ 0.5× budget with 0.2× for PQ and 0.1× for nav graph. |
| `--nav_L` | Search list size for the in-memory nav graph traversal that seeds the disk search. 0 disables the nav graph. Paper default: 100. |
| `--beamwidth_spike_ratio` | Look-ahead spike ratio α: fraction of the remaining unvisited pool to process in memory-first mode during the approach phase before deciding whether to issue I/O. Paper default: 0.25. |
| `--beam_decay_ratio` | Beam decay ratio β: in the convergence phase, the effective beam width is reduced by β each round to minimize delays in issuing I/O for vectors remaining in the final pool. Paper default: 0.95. |
| `--retset_capacity_ratio` | Candidate pool overflow ratio μ: the pool is allocated with μ × L capacity to accommodate extra candidates inserted by the priority pipeline without evicting active candidates. Paper default: 2.0. |
| `--cache_order_file` | Path to the page frequency order file produced by `reorder_pages_by_frequency` (only needed when the index has NOT been physically reordered, to guide cache loading order). |

**Example (SIFT100M, T=16 threads, with fsort):**
```bash
./build/apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix /mnt/nvme/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file /mnt/nvme/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort_gt.bin \
  -K 10 \
  -L 60 60 60 60 60 \
  -W 5 \
  -T 16 \
  --use_laann \
  --cache_ratio 0.172 \
  --nav_L 100 \
  --beamwidth_spike_ratio 0.25 \
  --beam_decay_ratio 0.95 \
  --retset_capacity_ratio 2.0
```

---

## Quick Reference: Paper Parameter Defaults

| Parameter | Value | Flag |
|---|---|---|
| Beam width | 5 | `-W` |
| Threads | 16 | `-T` |
| Spike ratio α | 0.25 | `--beamwidth_spike_ratio` |
| Decay ratio β | 0.95 | `--beam_decay_ratio` |
| Capacity ratio μ | 2.0 | `--retset_capacity_ratio` |
| Nav graph search depth | 100 | `--nav_L` |
| Nav graph degree | 24 | `-R` (build step) |
| Memory budget (100M) | 0.5× dataset | cache_ratio ≈ 0.172 |
| Memory budget (1B) | 0.3× dataset | cache_ratio ≈ 0.05 |
