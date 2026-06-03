#!/bin/bash
# PageANN search — generic template
#
# Usage:
#   Edit the CONFIGURE section below, then:
#   bash scripts/search_pageann_index.sh
#
# Output is printed to stdout and tee'd to a log file.

set -euo pipefail

# ============================================================
# CONFIGURE
# ============================================================

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" 2>/dev/null && pwd || echo "$HOME/DynaANN/build")"

# Index prefix produced by build_pageann_index.sh
INDEX_PREFIX="/path/to/PageANN/vamana_R<R>_L<L>_PQ<P>_PGD<N>_PageANN"

# Dataset
DATA_TYPE="uint8"    # float | int8 | uint8
DIST_FN="l2"
QUERY_FILE="/path/to/query.bin"
GT_FILE="/path/to/gt.bin"      # set to "null" to skip recall computation

# Search parameters
K=10                           # number of neighbors to return
L_VALUES="60 60 60 60 60"      # search list sizes (repeat for multiple trials)
W=5                            # beam width (I/O requests per round)
T=16                           # query threads

# PageANN-specific parameters (paper defaults)
CACHE_RATIO=0.172              # fraction of dataset to cache in DRAM
NAV_L=10                       # nav graph search depth (0 = disable nav graph)

# Optional: path to cache order file (leave empty to skip)
CACHE_ORDER_FILE="${INDEX_PREFIX}_cache_order.bin"

# Log output
LOG_DIR="$(dirname "${INDEX_PREFIX}")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/search_pageann_${TIMESTAMP}.log"

# ============================================================
# (nothing to edit below this line)
# ============================================================

SEARCH="${BUILD_DIR}/apps/search_disk_index"

echo "=== Pre-flight check ==="
[ -x "${SEARCH}" ] || { echo "ERROR: search_disk_index not found at ${SEARCH}"; exit 1; }
[ -f "${INDEX_PREFIX}.index" ] || { echo "ERROR: index file not found: ${INDEX_PREFIX}.index"; exit 1; }
[ -f "${QUERY_FILE}" ] || { echo "ERROR: query file not found: ${QUERY_FILE}"; exit 1; }
echo "  OK  ${SEARCH}"
echo "  OK  ${INDEX_PREFIX}.index"
echo "  OK  ${QUERY_FILE}"

# Resolve optional cache order file
CACHE_ORDER_ARG=""
if [ -f "${CACHE_ORDER_FILE}" ]; then
    echo "  OK  ${CACHE_ORDER_FILE} (cache order)"
    CACHE_ORDER_ARG="--cache_order_file ${CACHE_ORDER_FILE}"
fi

mkdir -p "${LOG_DIR}"
echo ""
echo "Index:          ${INDEX_PREFIX}"
echo "Threads:        T=${T}   W=${W}   L=${L_VALUES}"
echo "PageANN params: nav_L=${NAV_L}  cache_ratio=${CACHE_RATIO}"
echo "Log:            ${LOG_FILE}"
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
    ${CACHE_ORDER_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done. Results saved to: ${LOG_FILE}"
