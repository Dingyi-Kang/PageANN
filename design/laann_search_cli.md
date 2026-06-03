# LAANN Search: CLI Reference

This document describes the command-line parameters for running LAANN search via
`search_disk_index`, and when to use each option.

---

## Example Command

```bash
~/LAANN/build/apps/search_disk_index \
    --data_type uint8 \
    --dist_fn l2 \
    --index_path_prefix ~/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort \
    --query_file ~/sift100m/query.public.10K.u8bin \
    --gt_file ~/sift100m/LAANN/vamana_sift100M_R25_L150_PQ20_MGD450_LAANN_fsort_gt.bin \
    -K 10 \
    -L 30,50,70,100 \
    -W 5 \
    -T 16 \
    --use_laann \
    --cache_ratio 0.05 \
    --use_look_ahead_search \
    --persistence_window_width 5 \
    --nav_L 0
```

---

## Parameter Reference

### Required

| Flag | Meaning |
|------|---------|
| `--data_type` | Vector element type: `float`, `uint8`, or `int8` |
| `--dist_fn` | Distance function: `l2`, `mips`, or `cosine` |
| `--index_path_prefix` | Path prefix shared by all index files |
| `--query_file` | Query vectors in `.bin` format |
| `--gt_file` | Ground-truth file for recall evaluation |
| `-K` | Number of nearest neighbors to retrieve |
| `-L` | Comma-separated list of search list sizes to sweep |
| `--use_laann` | Must be set; selects LAANN search over PageANN |

### Core Search

| Flag | Default | Meaning |
|------|---------|---------|
| `-W` | 5 | Normal mode beam width: pages fetched per round (cached + disk) |
| `-T` | 16 | Number of search threads |

### Cache

| Flag | Default | Meaning |
|------|---------|---------|
| `--cache_ratio` | 0.0 | Fraction of all pages to preload into memory (preferred over `--num_pages_to_cache`). E.g. `0.05` = 5% of pages. Overrides `--num_pages_to_cache` when > 0. |
| `--num_pages_to_cache` | 0 | Absolute page count to cache. Use only when `cache_ratio` is not set. |

**Preference:** use `--cache_ratio` — it is portable across datasets of different sizes.
To convert from a memory budget: `cache_ratio = budget_bytes / (total_pages * 4096)`.

### Look-Ahead Mode

| Flag | Default | Meaning |
|------|---------|---------|
| `--use_look_ahead_search` | off | Enables look-ahead mode: each round collects cached pages first and defers disk I/O until the persistence check decides otherwise |
| `--persistence_window_width` | 0 | Top-N window scanned each round to test if the first skipped disk node is still prominent. `0` = always stay in look-ahead (no persistence check). Recommended: 5 |

See `laann_search_selection.md` for the full look-ahead / persistence-check design.

### Pipeline

| Flag | Default | Meaning |
|------|---------|---------|
| `--use_pipeline` | true | Set to `false` to disable async I/O pipeline. Pipeline overlaps disk reads with cached-page processing. Useful for controlled latency comparisons. |

### Navigation Graph

| Flag | Default | Meaning |
|------|---------|---------|
| `--nav_L` | 0 | Search depth on the nav graph for entry-point selection. `0` = disabled (use default medoid). Recommended when nav graph is built: 14. |

---

## Typical Configurations

| Goal | Key flags |
|------|-----------|
| Baseline (no cache, no look-ahead) | `--cache_ratio 0.0` |
| Cache only, normal beam search | `--cache_ratio 0.05` |
| Full look-ahead with persistence check | `--cache_ratio 0.05 --use_look_ahead_search --persistence_window_width 5` |
| Disable pipeline for latency comparison | add `--no_pipeline` |
| Enable nav graph entry point | `--nav_L 14` |
