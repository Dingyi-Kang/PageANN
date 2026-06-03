#!/bin/bash
# Search PageANN index on SIFT1M
#
# Run build_pageann_sift1m.sh first, then set INDEX_PREFIX below.
# Run from the DynaANN root:
#   bash sift1M/search_pageann_sift1m.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

# ---- Set these after running build_pageann_sift1m.sh ----
# The build script prints the exact INDEX_PREFIX to use — copy it here.
# Current build params: R=42, L=120, PQ=128, PGD=6 vectors/page
INDEX_PREFIX="${SCRIPT_DIR}/index/vamana_sift1M_R42_L120_PQ128_PGD6_PageANN"
GT_FILE="${SCRIPT_DIR}/sift_groundtruth.bin"
QUERY_FILE="${SCRIPT_DIR}/sift_query.bin"

# ---- Search parameters ----
DATA_TYPE="float"
DIST_FN="l2"
K=10
L_VALUES="40 50 60 70 80 100"   # multiple L values for recall–latency curve
W=5                              # beam width
T=16                             # query threads (adjust to your machine)

# ---- PageANN parameters ----
CACHE_RATIO=0.2   # cache ~20% of pages (~100 MB for 1M dataset)
NAV_L=10          # nav graph search depth

# ---- Output ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/index/search_pageann_${TIMESTAMP}.log"

# ============================================================

SEARCH="${BUILD_DIR}/apps/search_disk_index"

echo "=== Pre-flight check ==="
[ -x "${SEARCH}" ] || { echo "ERROR: search_disk_index not found at ${SEARCH}"; exit 1; }
[ -f "${INDEX_PREFIX}.index" ] || { echo "ERROR: index not found: ${INDEX_PREFIX}.index"; echo "  Run build_pageann_sift1m.sh first and update INDEX_PREFIX in this script."; exit 1; }
[ -f "${QUERY_FILE}" ] || { echo "ERROR: query file not found: ${QUERY_FILE}"; exit 1; }
echo "  OK  search_disk_index"
echo "  OK  ${INDEX_PREFIX}.index"
echo "  OK  ${QUERY_FILE}"

mkdir -p "$(dirname "${LOG_FILE}")"
echo ""
echo "Index  : ${INDEX_PREFIX}"
echo "Query  : ${QUERY_FILE}"
echo "K=${K}  W=${W}  T=${T}  L=${L_VALUES}"
echo "cache_ratio=${CACHE_RATIO}  nav_L=${NAV_L}"
echo "Log    : ${LOG_FILE}"
echo ""

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
    --cache_ratio "${CACHE_RATIO}" \
    --nav_L       "${NAV_L}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done. Results saved to: ${LOG_FILE}"
