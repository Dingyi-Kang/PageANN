# Paper Experiment Parameters

This document contains the exact parameters used in our paper experiments to reproduce the results.

**Memory Budget Categories:**
- **0%**: Minimal memory budget (0.06 GB, minimum memory budget for the program to run)
- **10%**: Low memory budget (~10% of dataset size)
- **20%**: Medium memory budget (~20% of dataset size)
- **30%**: High memory budget (~30% of dataset size)

**System Requirements:**
- **PageANN**: Can run under all memory budgets (0%, 10%, 20%, 30%)
- **DiskANN & Starling**: Require at least 10% memory budget
- **PipeANN & SPANN**: Require ≥ 30% memory budget

---

## SIFT 100M (uint8 × 128)

**Dataset Information:**
- Base file size: 11.9 GB
- Data type: uint8
- Dimensions: 128
- Max vectors per page: 32

**Memory Budget Mapping:**
- 0% = 0.06 GB
- 10% = 1.2 GB
- 20% = 2.4 GB
- 30% = 3.6 GB

### Experiment Construction Parameters

| System   | Memory Budget | Avg Vector Degree (R) | PQ Chunks | L   | Vectors per Page | Duplication |
|----------|---------------|-----------------------|-----------|-----|------------------|-------------|
| PageANN  | 0%            | 28                    | 12        | 150 | 7                | No          |
| PageANN  | 10%           | 28                    | 20        | 150 | 7                | No          |
| PageANN  | 20%           | 25                    | 20        | 150 | 18               | No          |
| PageANN  | 30%           | 25                    | 20        | 150 | 18               | No          |
| DiskANN  | 10%           | 23                    | 12        | 150 | 18               | No          |
| DiskANN  | 20%           | 23                    | 20        | 150 | 18               | No          |
| DiskANN  | 30%           | 23                    | 20        | 150 | 18               | No          |
| Starling | 10%           | 23                    | 12        | 150 | 18               | No          |
| Starling | 20%           | 23                    | 20        | 150 | 18               | No          |
| Starling | 30%           | 23                    | 20        | 150 | 18               | No          |
| PipeANN  | >30%          | 23                    | 20        | 150 | 18               | No          |
| SPANN    | 30%           | -                     | -         | -   | -                | Yes (2×)    |

---

### Experiment Search Parameters

**PageANN:**
- **20% (2.4 GB)**: Use sampled hash routing (sample ratio = 0.5), no page caching, search radius = 1
- **30% (3.6 GB)**: Use full hash routing (sample ratio = 1.0), cache pages until memory budget is filled, search radius = 1

**DiskANN & Starling:**
- Cache nodes until memory budget is filled

**PipeANN & SPANN:**
- No caching used (their minimum memory budget is already ≥30% of dataset size)


## SPACEV 100M (uint8 × 100)

**Dataset Information:**
- Base file size: 9.31 GB
- Data type: uint8
- Dimensions: 100
- Max vectors per page: 40

**Memory Budget Mapping:**
- 0% = 0.06 GB
- 10% = 1 GB
- 20% = 2 GB
- 30% = 3 GB

### Experiment Construction Parameters

| System   | Memory Budget | Avg Vector Degree (R) | PQ Chunks | L   | Vectors per Page | Duplication |
|----------|---------------|-----------------------|-----------|-----|------------------|-------------|
| PageANN  | 0%            | 25                    | 12        | 150 | 8                | No          |
| PageANN  | 10%           | 26                    | 18        | 150 | 8                | No          |
| PageANN  | 20%           | 26                    | 18        | 150 | 20               | No          |
| PageANN  | 30%           | 26                    | 18        | 150 | 20               | No          |
| DiskANN  | 10%           | 25                    | 10        | 150 | 20               | No          |
| DiskANN  | 20%           | 25                    | 18        | 150 | 20               | No          |
| DiskANN  | 30%           | 25                    | 18        | 150 | 20               | No          |
| Starling | 10%           | 25                    | 10        | 150 | 20               | No          |
| Starling | 20%           | 25                    | 18        | 150 | 20               | No          |
| Starling | 30%           | 25                    | 18        | 150 | 20               | No          |
| PipeANN  | >30%          | 25                    | 18        | 150 | 20               | No          |
| SPANN    | 30%           | -                     | -         | -   | -                | Yes (2×)    |

---

### Experiment Search Parameters

**PageANN:**
- **20% (2 GB)**: Use sampled hash routing (sample ratio = 0.056), no page caching, search radius = 2
- **30% (3 GB)**: Use sampled hash routing (sample ratio = 0.200), cache pages until memory budget is filled, search radius = 2

**DiskANN & Starling:**
- Cache nodes until memory budget is filled

**PipeANN & SPANN:**
- No caching used (their minimum memory budget is already ≥30% of dataset size)

## Deep 100M (float × 96)

**Dataset Information:**
- Base file size: 35.7 GB
- Data type: float
- Dimensions: 96
- Max vectors per page: 10

**Memory Budget Mapping:**
- 0% = 0.06 GB
- 10% = 3.6 GB
- 20% = 7.2 GB
- 30% = 10.8 GB (approximate)

### Experiment Construction Parameters

| System   | Memory Budget | Avg Vector Degree (R) | PQ Chunks | L   | Vectors per Page | Duplication |
|----------|---------------|-----------------------|-----------|-----|------------------|-------------|
| PageANN  | 0%            | 24                    | 36        | 150 | 3                | No          |
| PageANN  | 10%           | 31                    | 38        | 150 | 8                | No          |
| PageANN  | 20%           | 31                    | 60        | 150 | 8                | No          |
| PageANN  | 30%           | 31                    | 60        | 150 | 8                | No          |
| DiskANN  | 10%           | 31                    | 38        | 150 | 8                | No          |
| DiskANN  | 20%           | 31                    | 60        | 150 | 8                | No          |
| DiskANN  | 30%           | 31                    | 60        | 150 | 8                | No          |
| Starling | 10%           | 31                    | 38        | 150 | 8                | No          |
| Starling | 20%           | 31                    | 60        | 150 | 8                | No          |
| Starling | 30%           | 31                    | 60        | 150 | 8                | No          |
| PipeANN  | >30%          | 31                    | 60        | 150 | 8                | No          |
| SPANN    | 40%           | -                     | -         | -   | -                | Yes (2×)    |

---

### Experiment Search Parameters

**PageANN:**
- **20% (7.2 GB)**: Use sampled hash routing (sample ratio = 0.3), cache pages until memory budget is filled, search radius = 2
- **30% (10.8 GB)**: Use sampled hash routing (sample ratio = 0.3), cache pages until memory budget is filled, search radius = 2

**DiskANN & Starling:**
- Cache nodes until memory budget is filled

**PipeANN & SPANN:**
- No caching used (their minimum memory budget is already ≥30% of dataset size)

## SPACEV 1B (uint8 × 100)

**Dataset Information:**
- Base file size: 93.1 GB
- Data type: uint8
- Dimensions: 100
- Max vectors per page: 40

**Memory Budget Mapping:**
- 20% ≈ 20 GB
- >30% = >30 GB

### Experiment Construction Parameters

| System   | Memory Budget | Avg Vector Degree (R) | PQ Chunks | L   | Vectors per Page | Duplication |
|----------|---------------|-----------------------|-----------|-----|------------------|-------------|
| PageANN  | 20%           | 25                    | 21        | 150 | 20               | No          |
| DiskANN  | 20%           | 25                    | 21        | 150 | 20               | No          |
| PipeANN  | >30%          | 25                    | 21        | 150 | 20               | No          |

---

### Experiment Search Parameters

**PageANN:**
- **20%**: Use sampled hash routing (sample ratio = 0.2), cache pages until memory budget is filled, search radius = 1

**DiskANN:**
- Cache nodes until memory budget is filled

**PipeANN:**
- No caching used (its minimum memory budget is already ≥30% of dataset size)

## SIFT 1B (uint8 × 128)

**Dataset Information:**
- Base file size: 119 GB
- Data type: uint8
- Dimensions: 128
- Max vectors per page: 32

**Memory Budget Mapping:**
- 20% ≈ 24 GB
- >30% = >36 GB

### Experiment Parameters

| System   | Memory Budget | Avg Vector Degree (R) | PQ Chunks | L   | Vectors per Page | Duplication |
|----------|---------------|-----------------------|-----------|-----|------------------|-------------|
| PageANN  | 20%           | 24                    | 20        | 150 | 18               | No          |
| DiskANN  | 20%           | 23                    | 20        | 150 | 18               | No          |
| PipeANN  | 47%           | 23                    | 20        | 150 | 18               | No          |

---
### Experiment Search Parameters

**PageANN:**
- **20%**: Use sampled hash routing (sample ratio = 0.2), cache pages until memory budget is filled, search radius = 1

**DiskANN:**
- Cache nodes until memory budget is filled

**PipeANN:**
- No caching used (its minimum memory budget is already >30% of dataset size)


## Notes

- **Memory Budget**: Percentage of the dataset size allocated for in-memory structures
- **L**: Search list size parameter (candidate list size during beam search)
- **R**: Average vector degree in the graph
- **PQ Chunks**: Number of product quantization chunks for compression
- **Vectors per Page**: Number of vectors within each page
- **Duplication**: Whether the system uses vector duplication for improved recall
- **T**: Number of threads (T=16 for all experiments)
- **W**: Beamwidth (W=5 for all experiments)
- All experiments use L2 distance metric
- K=10 nearest neighbors are retrieved in all experiments
- "-" indicates parameter not applicable for that system
- PageANN is the only system capable of running under extreme low-memory budget (0%)
- **Memory Usage at 30%**: PageANN, DiskANN, Starling use page/node caching. Additionally, PageANN uses hash-based in-memory routing for entry point selection at 30% budget and Starling uses in-memory graph index for entry point selection.

**Index Construction Design:**

All systems are configured to maintain similar resource overhead:
1. **Disk Overhead Control**: Graph index size is maintained at approximately 2× the original dataset size across all systems
2. **Page Capacity Determination**: Based on the target disk overhead, we determine the number of vectors per page. The remaining page space is allocated for storing neighbor information
3. **Neighbor Information Storage**: Once vectors per page is fixed, all remaining space is filled with neighbor data. Because PageANN stores less metadata for neighbor data compared to baselines, it can fit more neighbor information in the same page space, resulting in slightly higher average vector degree (R) than baseline systems

Note: we can use command like below to monitor and control memory budget
sudo systemd-run --scope -p MemoryMax=0.1G /usr/bin/time -v ./search_disk_index ......


