# PageANN Paper Experiment Parameters

This document contains the exact parameters used in the PageANN paper experiments.

---

## Common Search Parameters

| Parameter | Value |
|---|---|
| K (neighbors returned) | 10 |
| W (beam width) | 5 |
| T (query threads) | 16 (throughput); also sweep T ∈ {2, 4, 8} |
| Nav graph search L (`--nav_L`) | 10 |
| Distance metric | L2 |

## Memory Budget Categories

- **0%**: Minimal (0.06 GB — minimum for program to run)
- **10%**: ~10% of dataset size
- **20%**: ~20% of dataset size
- **30%**: ~30% of dataset size

**System requirements:**
- **PageANN**: All memory budgets (0%, 10%, 20%, 30%)
- **DiskANN & Starling**: At least 10%
- **PipeANN & SPANN**: ≥ 30%

---

## SIFT100M (uint8 × 128)

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

---

## SPACEV100M (int8 × 100)

**Memory Budget Mapping:** 0% = 0.06 GB · 10% = 1 GB · 20% = 2 GB · 30% = 3 GB

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN | 0% | 25 | 12 | 150 | 8 |
| PageANN | 10–30% | 26 | 18 | 150 | 8–20 |
| DiskANN | 10–30% | 25 | 10–18 | 150 | — |
| Starling | 10–30% | 25 | 10–18 | 150 | 20 |
| PipeANN | >30% | 25 | 18 | 150 | 20 |

---

## DEEP100M (float × 96)

**Memory Budget Mapping:** 10% = 3.6 GB · 20% = 7.2 GB · 30% = 10.8 GB

| System | Budget | R | PQ Chunks | L | Vectors/Page |
|---|---|---|---|---|---|
| PageANN | 10–30% | 31 | 38–60 | 150 | 8 |
| DiskANN | 10–30% | 31 | 38–60 | 150 | — |
| Starling | 10–30% | 31 | 38–60 | 150 | 8 |
| PipeANN | >30% | 31 | 60 | 150 | 8 |

---

## SPACEV1B / SIFT1B

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
- **enable_spare_fill**: all PageANN disk indexes were built with spare-space filling enabled (the default). To disable: `--enable_spare_fill false` in `generate_page_graph`.
- Monitor and cap memory usage with: `sudo systemd-run --scope -p MemoryMax=6G ./build/apps/search_disk_index ...`
