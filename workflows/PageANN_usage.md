**Usage for PageANN Indices**
===============================

PageANN is a page-based approximate nearest neighbor search system that organizes vectors into disk pages for efficient SSD-based retrieval. The workflow involves: (1) recommending graph parameters, (2) building a Vamana index, (3) converting to a page-based graph, (4) optionally generating hash buckets for routing, and (5) searching the index.

## Step 1: Recommend Graph Degree Parameters
Use `apps/recommend_vamana_graph_degree` to determine optimal graph degree based on memory budget.
---------------------------------------------------------------------------------------------------------

**Arguments:**

1. **--data_path**: Input dataset in binary format (same format as SSD index).
2. **--data_type**: Data type - float (32-bit), int8 (signed 8-bit), or uint8 (unsigned 8-bit).
3. **--full_ooc**: Set to `true` for fully out-of-core processing with minimum memory overhead.
4. **--num_PQ_chunks**: Number of PQ (Product Quantization) chunks for compression (typically 12-32).
5. **--mem_budget_in_GB**: Memory budget in GB for the searching process.
6. **--min_degree_per_node**: Minimum graph degree per node (output parameter recommendation).

**Example:**
```bash
./apps/recommend_vamana_graph_degree \
  --data_path ~/sift100m/learn.100M.u8bin \
  --data_type uint8 \
  --full_ooc false \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 3.6 \
  --min_degree_per_node 23
```

## Step 2: Build Vamana Disk Index
Use `apps/build_vamana_disk_index` to construct the initial Vamana graph.
---------------------------------------------------------------------------

**Arguments:**

1. **--data_type**: Data type (must match Step 1).
2. **--dist_fn**: Distance function - `l2` (Euclidean), `cosine`, or `mips` (max inner product).
3. **--data_path**: Input dataset in binary format.
4. **--index_path_prefix**: Output prefix for index files.
5. **-R (--max_degree)**: Maximum graph degree
6. **-L (--Lbuild)**: Build-time search list size (e.g., 100-200). Higher L → better quality, slower build.
7. **-B (--search_DRAM_budget)**: Search-time memory budget in GB.
8. **-M (--build_DRAM_budget)**: Build-time memory budget in GB.

**Example:**
```bash
./apps/build_vamana_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --index_path_prefix /mnt/disk/PageANN/vamana_sift100M_R25_L150_PQ20 \
  -R 25 \
  -L 150 \
  -B 2.2 \
  -M 40
```

## Step 3: Generate Page-Based Graph
Use `apps/generate_page_graph` to convert Vamana index into PageANN format.
-----------------------------------------------------------------------------

**Arguments:**

1. **--data_type**: Data type (must match previous steps).
2. **--dist_fn**: Distance function (must match Step 2).
3. **--data_path**: Original dataset path.
4. **--vamana_index_path_prefix**: Prefix from Step 2 output.
5. **--R**: Graph degree (must match Step 2).
6. **--num_PQ_chunks**: PQ chunks (must match Step 2).
7. **--mem_budget_in_GB**: Memory budget for search process.
8. **--full_ooc**: Fully out-of-core processing flag (`true` for minimum memory overhead).
9. **--min_degree_per_node**: Minimum degree from Step 1 recommendation.

**Example:**
```bash
./apps/generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --vamana_index_path_prefix /mnt/disk/PageANN/vamana_sift100M_R25_L150_PQ20 \
  --R 25 \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 36.0 \
  --full_ooc false \
  --min_degree_per_node 23
```

**Output:** PageANN index with suffix `_PGDxxx_PageANN` (xxx = nodes per page).

## Step 4: Compute Ground Truth
Use `apps/utils/compute_groundtruth` to generate recall evaluation data.
--------------------------------------------------------------------------

### Option A: Using Pre-computed Vamana Ground Truth (Faster)

**Arguments:**

1. **--data_type**: Data type.
2. **--dist_fn**: Distance function.
3. **--base_file**: Base dataset file.
4. **--query_file**: Query dataset file.
5. **--index_prefix**: PageANN index prefix (from Step 3).
6. **--gt_file**: Output ground truth file path.
7. **--vamana_gt_file**: Pre-computed Vamana ground truth (optional, for acceleration).
8. **--K**: Number of nearest neighbors for ground truth.

**Example:**
```bash
./apps/utils/compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file ~/sift100m/learn.100M.u8bin \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --index_prefix /mnt/disk/PageANN/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --gt_file /mnt/disk/PageANN/gt_PageANN_K100 \
  --vamana_gt_file ~/sift100m/gt_DiskANN_K100 \
  --K 100
```

### Option B: Computing Ground Truth from Scratch

Omit `--vamana_gt_file` to compute ground truth directly:

**Example:**
```bash
./apps/utils/compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file ~/sift100m/learn.100M.u8bin \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --index_prefix /mnt/disk/PageANN/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --gt_file /mnt/disk/PageANN/gt_PageANN_K100 \
  --K 100
```

## Step 5: Search the PageANN Index
Use `apps/search_disk_index` with various routing and caching strategies.
---------------------------------------------------------------------------

### Common Search Arguments:

1. **--data_type**: Data type.
2. **--dist_fn**: Distance function.
3. **--index_path_prefix**: PageANN index prefix (from Step 3).
4. **--query_file**: Query dataset file.
5. **--gt_file**: Ground truth file (from Step 4).
6. **-K**: Number of neighbors to retrieve.
7. **-L**: Search list size (candidate pool). Higher L → better recall, slower search.
8. **-W (--beamwidth)**: Beam width for parallel search (1-8).
9. **-T (--num_threads)**: Number of search threads.

### Strategy 1: Basic Search with Medoid Entry Points

Standard search using medoid-based entry point selection:

```bash
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  -W 5 \
  -T 16
```

### Strategy 2: Hash-Based Routing (Full Hash Buckets)

Use hash-based entry point selection with all vectors in hash buckets:

**Additional Arguments:**
- **--use_hash_routing**: Enable hash-based routing (`true`/`false`).
- **--radius**: Hamming radius for hash bucket expansion (0-2).

```bash
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  --use_hash_routing true \
  --radius 1 \
  -W 5 \
  -T 16
```

**Prerequisite:** Generate hash buckets from all vectors (see Step 5a below).

### Strategy 3: Sampled Hash-Based Routing

Use hash-based routing with sampled vectors (lower memory footprint):

**Additional Arguments:**
- **--use_sampled_hash_routing**: Enable sampled hash routing (`true`/`false`).

```bash
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  --use_sampled_hash_routing true \
  --radius 1 \
  --num_pages_to_cache 0 \
  -W 5 \
  -T 16
```

**Prerequisite:** Generate sampled hash buckets (see Step 5b below).

### Strategy 4: Hash Routing + Page Caching

Combine hash-based routing with page caching for optimal performance:

```bash
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  --use_hash_routing true \
  --radius 1 \
  --num_pages_to_cache 10000 \
  -W 5 \
  -T 16
```

## Step 5a: Generate Hash Buckets (Full)
Use `apps/utils/generate_hash_buckets` to create hash buckets from all vectors.
---------------------------------------------------------------------------------

**Arguments:**

1. **--data_type**: Data type (int8 or uint8).
2. **--graph_index_prefix**: PageANN index prefix.
3. **--data_file_to_use**: Original dataset file.
4. **--compute_from_beginning**: Set `true` to compute projection matrix from scratch.
5. **--sample_ratio**: Ratio of vectors to include (1.0 = all vectors).
6. **--min_num_per_bucket**: Minimum number of vectors per bucket.

**Example:**
```bash
./apps/utils/generate_hash_buckets \
  --data_type int8 \
  --graph_index_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --data_file_to_use ~/sift100m/learn.100M.u8bin \
  --compute_from_beginning true \
  --sample_ratio 1.0 \
  --min_num_per_bucket 1
```

**Output:** `<index_prefix>_buckets.bin` and `<index_prefix>_projection_matrix_file.bin`

## Step 5b: Generate Hash Buckets (Sampled)
Generate sampled hash buckets to reduce memory overhead.
----------------------------------------------------------

**Arguments:** Same as Step 5a, but:
- **--compute_from_beginning**: Set `false` to reuse existing projection matrix and buckets.
- **--sample_ratio**: Fraction of vectors to sample (e.g., 0.5 = 50%).

**Example:**
```bash
./apps/utils/generate_hash_buckets \
  --data_type int8 \
  --graph_index_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --data_file_to_use ~/sift100m/learn.100M.u8bin \
  --compute_from_beginning false \
  --sample_ratio 0.5 \
  --min_num_per_bucket 1
```

**Output:** `<index_prefix>_subset_buckets.bin`

## Step 6: Regenerate PQ with Different Compression (Optional)
Regenerate Product Quantization (PQ) compressed data with a different number of chunks for an existing PageANN index.
-------------------------------------------------------------------------------------------------------------------------

This utility allows you to change the PQ compression level (number of chunks) without rebuilding the entire index. Useful for:
- Experimenting with different memory/accuracy trade-offs
- Adjusting search-time memory budget after index construction
- Creating multiple PQ variants of the same index

**Use Case:** You built a PageANN index with 20 PQ chunks but want to try 12 or 16 chunks for lower memory usage.

### Arguments:

1. **<data_type>**: Data type (`float`, `int8`, or `uint8`)
2. **<data_file>**: Original dataset file (`.bin` format)
3. **<PQ_prefix_path>**: PageANN index prefix (from Step 3)
4. **<target_pq_chunks>**: New number of PQ chunks (e.g., 12, 16, 20, 24)

### Example:

Regenerate PQ with 16 chunks for an existing index:

```bash
./apps/utils/generate_reorder_pq \
  uint8 \
  ~/sift100m/learn.100M.u8bin \
  ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  16
```

### Output Files:

- `<index_prefix>_PQ16_pq_pivots.bin` - PQ codebook with 16 chunks
- `<index_prefix>_PQ16_reorder_pq_compressed.bin` - Reordered PQ data (16 bytes per vector)

### Usage in Search:

To use the newly generated PQ data during search, specify the PQ path prefix:

```bash
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --pq_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN_PQ16 \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  -W 5 \
  -T 16
```

### Notes:

- **PQ chunks trade-off**:
  - Fewer chunks (e.g., 12) = Lower memory, lower accuracy
  - More chunks (e.g., 24) = Higher memory, higher accuracy
- The graph structure remains unchanged; only PQ compression changes
- No need to rebuild the entire index
- Typical values: 12, 16, 20, 24, 32 chunks

---

## Complete Workflow Example: SIFT-100M

```bash
# Step 1: Recommend graph degree
./apps/recommend_vamana_graph_degree \
  --data_path ~/sift100m/learn.100M.u8bin \
  --data_type uint8 \
  --full_ooc false \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 3.6 \
  --min_degree_per_node 23

# Step 2: Build Vamana index
./apps/build_vamana_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20 \
  -R 25 \
  -L 150 \
  -B 2.2 \
  -M 40

# Step 3: Generate PageANN graph
./apps/generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path ~/sift100m/learn.100M.u8bin \
  --vamana_index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20 \
  --R 25 \
  --num_PQ_chunks 20 \
  --mem_budget_in_GB 36.0 \
  --full_ooc false \
  --min_degree_per_node 23

# Step 4: Compute ground truth
./apps/utils/compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file ~/sift100m/learn.100M.u8bin \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --index_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  --K 100

# Step 5a: Generate hash buckets (optional, for hash routing)
./apps/utils/generate_hash_buckets \
  --data_type int8 \
  --graph_index_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --data_file_to_use ~/sift100m/learn.100M.u8bin \
  --compute_from_beginning true \
  --sample_ratio 1.0 \
  --min_num_per_bucket 1

# Step 6: Search with hash routing and caching
./apps/search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix ~/sift100m/vamana_sift100M_R25_L150_PQ20_PGD447_PageANN \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file ~/sift100m/gt_PageANN_K100 \
  -K 10 \
  -L 76 \
  --use_hash_routing true \
  --radius 1 \
  --num_pages_to_cache 100 \
  -W 5 \
  -T 16
```

## Demo of Results

**SEARCH PERFORMANCE RESULTS**

| L  | Beamwidth | QPS    | Latency (us) | IO (us) | Other (us) | Mean IOs | Hops | Recall@10 | Recall@5 | Recall@2 | Recall@1 |
|----|-----------|--------|--------------|---------|------------|----------|------|-----------|----------|----------|----------|
| 76 | 5         | 2154.5 | 7392.21      | 6000.49 | 1391.72    | 77.2     | 16.2 | 90.1290   | 92.5140  | 94.1300  | 94.7700  |

---

## Key Differences from Standard SSD Index

1. **Page-Based Organization**: PageANN groups multiple vectors per disk page for efficient I/O.
2. **Hash-Based Routing**: Optional hash-based entry point selection (vs. medoid-only in SSD index).
3. **Page Caching**: Cache entire pages rather than individual nodes.
4. **Two-Stage Construction**: Vamana → PageANN conversion with degree constraints.

---

## Performance Notes

- **use_hash_routing vs. use_sampled_hash_routing**:
  - `use_hash_routing` loads all vectors into hash buckets (higher memory, better recall)
  - `use_sampled_hash_routing` samples vectors (lower memory, slightly lower recall)
  - Both cannot be true simultaneously

- **Radius Parameter**: Controls Hamming distance for hash bucket expansion
  - radius=0: Exact hash match only
  - radius=1: Include neighbors within Hamming distance 1 (recommended)
  - radius=2: Further expansion (higher recall, more computation)

- **Page Caching**: Cache 10K-100K pages for frequently-accessed data based on available RAM

