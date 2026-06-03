# Paper Experiment Parameters

This document contains the exact parameters used in our paper experiments to reproduce the results.

---

## LAANN Paper Experiments

### Hardware

- CPU: Intel Core i9-13900 (2.0 GHz base, 24 cores), 128 GB DDR5 RAM
- SSD: 1 TB KIOXIA NVMe
- OS: Ubuntu 24.04

### Common Search Parameters (All Datasets)

| Parameter | Value |
|---|---|
| K (neighbors returned) | 10 |
| W (beam width) | 5 |
| T (query threads) | 16 (throughput); also sweep T ∈ {2, 4, 8} |
| Spike ratio α (`--beam_spike_ratio`) | 0.25 |
| Decay ratio β (`--decay_ratio`) | 0.95 |
| Nav graph search L (`--nav_L`) | 100 |
| Distance metric | L2 |

### Memory Budget Allocation (100M-Scale, 0.5× dataset size)

| Component | Fraction | Systems |
|---|---|---|
| Compressed vectors (PQ) | 0.2× | All |
| Cached disk pages | 0.2× | LAANN, PageANN, DiskANN (0.3×), Starling, MARGO |
| In-memory nav graph | 0.1× | LAANN, PageANN, Starling, MARGO |
| In-memory graph (full vectors) | 0.3× | PipeANN (no cached pages) |

### Memory Budget Allocation (Billion-Scale, 0.3× dataset size)

| Component | Fraction | Systems |
|---|---|---|
| Compressed vectors (PQ) | 0.2× | All |
| Cached disk pages | 0.05× | LAANN only |
| In-memory nav graph | 0.05× | LAANN only |
| In-memory graph (full vectors) | 0.1× | PageANN, PipeANN |
| Cached disk pages | 0.1× | DiskANN (no in-memory graph) |

---

### SIFT100M (uint8 × 128)

**Dataset:** 100M vectors, 128 dimensions, uint8  
**File size:** 11.9 GB | **Memory budget (0.5×):** ~5.95 GB | **Cache ratio:** ~0.172

#### Construction Parameters

| System | R | PQ Chunks | L | Vectors/Page | Nav Graph R | Nav Graph L |
|---|---|---|---|---|---|---|
| LAANN | 25 | 20 | 150 | 18 | 24 | 150 |
| PageANN | 25 | 20 | 150 | 18 | 24 | 150 |
| DiskANN | 23 | 20 | 150 | — | — | — |
| Starling | 23 | 20 | 150 | 18 | 23 | 150 |
| MARGO | 23 | 20 | 150 | 18 | 23 | 150 |
| PipeANN | 23 | 20 | 150 | 18 | 23 | 150 |

#### Search Command (LAANN)

```bash
./build/apps/search_disk_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix ~/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort \
  --disk_file_path   ~/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort.index \
  --mem_index_path   ~/sift100m/LAANN/MEM_R_24_L_150_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.005_FREQ_RATE0.01/_index \
  --query_file ~/sift100m/query.public.10K.u8bin \
  --gt_file    ~/sift100m/sift100m_gt_K100 \
  --result_path ~/sift100m/results/ \
  -K 10 -L 60 60 60 60 60 -W 5 -T 16 \
  --use_page_search 1 \
  --cache_ratio 0.172 \
  --beam_spike_ratio 0.25 --decay_ratio 0.95 --nav_L 100
```

---

### SPACEV100M (int8 × 100)

**Dataset:** 100M vectors, 100 dimensions, int8  
**File size:** 9.31 GB | **Memory budget (0.5×):** ~4.66 GB | **Cache ratio:** ~0.172

#### Construction Parameters

| System | R | PQ Chunks | L | Vectors/Page | Nav Graph R | Nav Graph L |
|---|---|---|---|---|---|---|
| LAANN | 25 | 18 | 150 | 20 | 24 | 150 |
| PageANN | 25 | 18 | 150 | 20 | 24 | 150 |
| DiskANN | 25 | 18 | 150 | — | — | — |
| Starling | 25 | 18 | 150 | 20 | 23 | 150 |
| MARGO | 25 | 18 | 150 | 20 | 23 | 150 |
| PipeANN | 25 | 18 | 150 | 20 | 23 | 150 |

#### Search Command (LAANN)

```bash
./build/apps/search_disk_index \
  --data_type int8 --dist_fn l2 \
  --index_path_prefix ~/spacev100m/LAANN/vamana_spacev100M_R25_L150_PQ18_MGD_LAANN_fsort \
  --disk_file_path   ~/spacev100m/LAANN/vamana_spacev100M_R25_L150_PQ18_MGD_LAANN_fsort.index \
  --mem_index_path   ~/spacev100m/LAANN/MEM_R_24_L_150_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.005_FREQ_RATE0.01/_index \
  --query_file ~/spacev100m/query.i8bin \
  --gt_file    ~/spacev100m/spacev100m_gt_K100 \
  --result_path ~/spacev100m/results/ \
  -K 10 -L 60 60 60 60 60 -W 5 -T 16 \
  --use_page_search 1 \
  --cache_ratio 0.172 \
  --beam_spike_ratio 0.25 --decay_ratio 0.95 --nav_L 100
```

---

### DEEP100M (float × 96)

**Dataset:** 100M vectors, 96 dimensions, float  
**File size:** 35.7 GB | **Memory budget (0.5×):** ~17.85 GB | **Cache ratio:** ~0.172

#### Construction Parameters

| System | R | PQ Chunks | L | Vectors/Page | Nav Graph R | Nav Graph L |
|---|---|---|---|---|---|---|
| LAANN | 31 | 60 | 150 | 8 | 24 | 150 |
| PageANN | 31 | 60 | 150 | 8 | 24 | 150 |
| DiskANN | 31 | 60 | 150 | — | — | — |
| Starling | 31 | 60 | 150 | 8 | 23 | 150 |
| MARGO | 31 | 60 | 150 | 8 | 23 | 150 |
| PipeANN | 31 | 60 | 150 | 8 | 23 | 150 |

---

### SIFT1B (uint8 × 128)

**Dataset:** 1B vectors, 128 dimensions, uint8  
**File size:** 119 GB | **Memory budget (0.3×):** ~35.7 GB

Starling and MARGO excluded (exceed memory capacity).

#### Construction Parameters

| System | R | PQ Chunks | L | Vectors/Page | Nav Graph R | Nav Graph L |
|---|---|---|---|---|---|---|
| LAANN | 25 | 20 | 150 | 18 | 24 | 150 |
| PageANN | 25 | 20 | 150 | 18 | 24 | 150 |
| DiskANN | 23 | 20 | 150 | — | — | — |
| PipeANN | 23 | 20 | 150 | 18 | 23 | 150 |

#### Search Command (LAANN)

```bash
./build/apps/search_disk_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix ~/sift1b/LAANN/vamana_sift1B_R25_L150_PQ20_MGD_LAANN_fsort \
  --disk_file_path   ~/sift1b/LAANN/vamana_sift1B_R25_L150_PQ20_MGD_LAANN_fsort.index \
  --mem_index_path   ~/sift1b/LAANN/MEM_R_24_L_150_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.005_FREQ_RATE0.01/_index \
  --query_file ~/sift1b/query.public.10K.u8bin \
  --gt_file    ~/sift1b/sift1b_gt_K100 \
  --result_path ~/sift1b/results/ \
  -K 10 -L 60 60 60 60 60 -W 5 -T 16 \
  --use_page_search 1 \
  --cache_ratio 0.05 \
  --beam_spike_ratio 0.25 --decay_ratio 0.95 --nav_L 100
```

---

### SPACEV1B (int8 × 100)

**Dataset:** 1B vectors, 100 dimensions, int8  
**File size:** 93.1 GB | **Memory budget (0.3×):** ~27.9 GB

#### Construction Parameters

| System | R | PQ Chunks | L | Vectors/Page | Nav Graph R | Nav Graph L |
|---|---|---|---|---|---|---|
| LAANN | 25 | 21 | 150 | 20 | 24 | 150 |
| PageANN | 25 | 21 | 150 | 20 | 24 | 150 |
| DiskANN | 25 | 21 | 150 | — | — | — |
| PipeANN | 25 | 21 | 150 | 20 | 23 | 150 |

#### Search Command (LAANN)

```bash
./build/apps/search_disk_index \
  --data_type int8 --dist_fn l2 \
  --index_path_prefix ~/spacev1b/LAANN/vamana_spacev1B_R25_L150_PQ21_MGD_LAANN_fsort \
  --disk_file_path   ~/spacev1b/LAANN/vamana_spacev1B_R25_L150_PQ21_MGD_LAANN_fsort.index \
  --mem_index_path   ~/spacev1b/LAANN/MEM_R_24_L_150_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.005_FREQ_RATE0.01/_index \
  --query_file ~/spacev1b/query.i8bin \
  --gt_file    ~/spacev1b/spacev1b_gt_K100 \
  --result_path ~/spacev1b/results/ \
  -K 10 -L 60 60 60 60 60 -W 5 -T 16 \
  --use_page_search 1 \
  --cache_ratio 0.05 \
  --beam_spike_ratio 0.25 --decay_ratio 0.95 --nav_L 100
```

---

### Performance Results at Recall@10 = 0.9

**100M-scale (0.5× memory budget, T=16):**

| Dataset | LAANN QPS | LAANN Latency (ms) | LAANN Mean I/Os | vs. PageANN QPS |
|---|---|---|---|---|
| SIFT100M | 3825.6 | 4.17 | 36.1 | 1.41× |
| SPACEV100M | 7987.2 | 1.99 | 15.7 | 2.12× |
| DEEP100M | 9516.9 | 1.67 | 11.4 | 1.96× |

**Billion-scale (0.3× memory budget, T=16):**

| Dataset | LAANN QPS | LAANN Latency (ms) | vs. PageANN QPS |
|---|---|---|---|
| SIFT1B | 1147.7 | 13.87 | 1.42× |
| SPACEV1B | (see paper) | (see paper) | >2× |

---

## PageANN Paper Experiments (Legacy)

The following parameters are from the original PageANN paper and are preserved for reference.

**Memory Budget Categories:**
- **0%**: Minimal (0.06 GB — minimum for program to run)
- **10%**: ~10% of dataset size
- **20%**: ~20% of dataset size
- **30%**: ~30% of dataset size

**System Requirements:**
- **PageANN**: All memory budgets (0%, 10%, 20%, 30%)
- **DiskANN & Starling**: At least 10%
- **PipeANN & SPANN**: ≥ 30%

### SIFT100M

**Memory Budget Mapping:** 0% = 0.06 GB · 10% = 1.2 GB · 20% = 2.4 GB · 30% = 3.6 GB

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN | 0% | 28 | 12 | 150 | 7 |
| PageANN | 10% | 28 | 20 | 150 | 7 |
| PageANN | 20% | 25 | 20 | 150 | 18 |
| PageANN | 30% | 25 | 20 | 150 | 18 |
| DiskANN | 10–30% | 23 | 12–20 | 150 | — |
| Starling | 10–30% | 23 | 12–20 | 150 | 18 |
| PipeANN | >30% | 23 | 20 | 150 | 18 |

### SPACEV100M

**Memory Budget Mapping:** 0% = 0.06 GB · 10% = 1 GB · 20% = 2 GB · 30% = 3 GB

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN | 0% | 25 | 12 | 150 | 8 |
| PageANN | 10–30% | 26 | 18 | 150 | 8–20 |
| DiskANN | 10–30% | 25 | 10–18 | 150 | — |
| Starling | 10–30% | 25 | 10–18 | 150 | 20 |
| PipeANN | >30% | 25 | 18 | 150 | 20 |

### DEEP100M

**Memory Budget Mapping:** 10% = 3.6 GB · 20% = 7.2 GB · 30% = 10.8 GB

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN | 10–30% | 31 | 38–60 | 150 | 8 |
| DiskANN | 10–30% | 31 | 38–60 | 150 | — |
| Starling | 10–30% | 31 | 38–60 | 150 | 8 |
| PipeANN | >30% | 31 | 60 | 150 | 8 |

### SPACEV1B / SIFT1B

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN SPACEV1B | 20% | 25 | 21 | 150 | 20 |
| PageANN SIFT1B | 20% | 24 | 20 | 150 | 18 |
| DiskANN SIFT1B | 20% | 23 | 20 | 150 | 18 |

---

## Notes

- **W**: Beamwidth = 5 for all experiments
- **T**: Threads = 16 for all throughput experiments
- **K**: 10 nearest neighbors retrieved in all experiments
- All experiments use L2 distance metric
- **cache_ratio**: fraction of dataset cached in DRAM at search time
- **enable_spare_fill**: all PageANN and LAANN disk indexes were built with spare-space filling enabled (the default). To disable: `--enable_spare_fill false` in `generate_page_graph`.
- Monitor and cap memory usage with: `sudo systemd-run --scope -p MemoryMax=6G ./build/apps/search_disk_index ...`
