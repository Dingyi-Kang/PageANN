#!/bin/bash
# Test nav graph built with page subsampling (--num_sampled_pages).
# Builds a nav graph using 100000 evenly-distributed pages out of ~166667 total,
# then runs search and prints recall + latency.
#
# Run from the DynaANN root:
#   bash sift1M/test_nav_page_sampling.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

# ---- Index (fsort) ----
LAANN_INDEX="${SCRIPT_DIR}/index/vamana_sift1M_R42_L120_PQ128_MGD252_LAANN_fsort.index"
INDEX_PREFIX="${LAANN_INDEX%.index}"
GT_FILE="${SCRIPT_DIR}/index/vamana_sift1M_R42_L120_PQ128_MGD252_LAANN_fsort_gt.bin"
QUERY_FILE="${SCRIPT_DIR}/sift_query.bin"

# ---- Nav graph build params ----
DATA_TYPE="float"
DIST_FN="l2"
NAV_R=24
NAV_L=100
NAV_ALPHA=1.2
NAV_THREADS=16
NUM_SAMPLED_PAGES=100000   # pages to sample out of ~166667 total

# ---- Search params ----
K=10
L_VALUES="60 80 100"
W=5
T=8
CACHE_RATIO=0.2
DECAY_RATIO=0.95
RETSET_RATIO=2.0
NAV_L_SEARCH=100

# ---- Binaries ----
BUILD_NAV="${BUILD_DIR}/apps/utils/build_laann_nav_graph"
SEARCH="${BUILD_DIR}/apps/search_disk_index"

# ---- Pre-flight ----
echo "=== Pre-flight check ==="
[ -x "${BUILD_NAV}" ] || { echo "ERROR: build_laann_nav_graph not found"; exit 1; }
[ -x "${SEARCH}" ]    || { echo "ERROR: search_disk_index not found"; exit 1; }
[ -f "${LAANN_INDEX}" ] || { echo "ERROR: LAANN index not found: ${LAANN_INDEX}"; exit 1; }
[ -f "${QUERY_FILE}" ]  || { echo "ERROR: query file not found: ${QUERY_FILE}"; exit 1; }
[ -f "${GT_FILE}" ]     || { echo "ERROR: GT file not found: ${GT_FILE}"; exit 1; }
echo "All checks passed."
echo ""

# ---- Step 1: Build nav graph with page subsampling ----
echo "=== Building nav graph: ${NUM_SAMPLED_PAGES} sampled pages ==="
"${BUILD_NAV}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --laann_disk_index_file "${LAANN_INDEX}" \
    -R "${NAV_R}" \
    -L "${NAV_L}" \
    --alpha "${NAV_ALPHA}" \
    -T "${NAV_THREADS}" \
    --num_sampled_pages "${NUM_SAMPLED_PAGES}"

echo ""

# ---- Step 2: Search ----
echo "=== Search (K=${K}, W=${W}, T=${T}, L=${L_VALUES}) ==="
# shellcheck disable=SC2086
"${SEARCH}" \
    --data_type         "${DATA_TYPE}" \
    --dist_fn           "${DIST_FN}" \
    --index_path_prefix "${INDEX_PREFIX}" \
    --query_file        "${QUERY_FILE}" \
    --gt_file           "${GT_FILE}" \
    -K "${K}" \
    -L ${L_VALUES} \
    -W "${W}" \
    -T "${T}" \
    --use_laann \
    --cache_ratio            "${CACHE_RATIO}" \
    --nav_L                  "${NAV_L_SEARCH}" \
    --beam_decay_ratio       "${DECAY_RATIO}" \
    --retset_capacity_ratio  "${RETSET_RATIO}"
