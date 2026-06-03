#!/bin/bash
# Build PageANN index for SIFT1M
#
# Steps:
#   0. Convert fvecs/ivecs → bin format
#   1. Build Vamana disk index
#   2. Generate PageANN page graph
#   3. Build in-memory nav graph
#
# Run from the DynaANN root:
#   bash sift1M/build_pageann_sift1m.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
DATA_DIR="${SCRIPT_DIR}"
INDEX_DIR="${SCRIPT_DIR}/index"
LOG_DIR="${INDEX_DIR}/build_logs"
mkdir -p "${INDEX_DIR}" "${LOG_DIR}"

# ---- Binaries ----
FVECS_TO_BIN="${BUILD_DIR}/apps/utils/fvecs_to_bin"
IVECS_TO_BIN="${BUILD_DIR}/apps/utils/ivecs_to_bin"
BUILD_VAMANA="${BUILD_DIR}/apps/build_vamana_disk_index"
GEN_PAGE="${BUILD_DIR}/apps/generate_page_graph"
BUILD_NAV="${BUILD_DIR}/apps/utils/build_pageann_nav_graph"

# ---- Dataset ----
DATA_TYPE="float"
DIST_FN="l2"
BASE_FVECS="${DATA_DIR}/sift_base.fvecs"
QUERY_FVECS="${DATA_DIR}/sift_query.fvecs"
GT_IVECS="${DATA_DIR}/sift_groundtruth.ivecs"
BASE_BIN="${DATA_DIR}/sift_base.bin"
QUERY_BIN="${DATA_DIR}/sift_query.bin"
GT_BIN="${DATA_DIR}/sift_groundtruth.bin"

# ---- Vamana build params ----
# SIFT1M: 1M × 128-dim float (~512 MB raw data)
# R should match the page-level degree per vector so Vamana fills the page at
# full quality without heavy reliance on the spare-fill phase.
# Page capacity = 6 vectors/page, page degree = 252 (42×6) → R = 252/6 = 42.
VAMANA_R=42
VAMANA_L=120
VAMANA_B=0.5   # search-time memory budget in GB (≈ 0.5× dataset size)
VAMANA_M=4     # build-time memory budget in GB
N_POINTS=1000000
DIM=128

# PQ_CHUNKS: derived the same way as build_vamana_disk_index:
#   PQ_CHUNKS = min(dim, floor(B_bytes / N_points))
PQ_CHUNKS=$(python3 -c "import math; print(min(${DIM}, math.floor(${VAMANA_B}*1024**3/${N_POINTS})))")
echo "Derived PQ_CHUNKS=${PQ_CHUNKS} from B=${VAMANA_B} GB, N=${N_POINTS}, dim=${DIM}"
VAMANA_PREFIX="${INDEX_DIR}/vamana_sift1M_R${VAMANA_R}_L${VAMANA_L}_PQ${PQ_CHUNKS}"

# ---- PageANN page graph params ----
MIN_DEGREE=${VAMANA_R}   # set equal to R for full page utilization

# ---- Nav graph params ----
NAV_R=23
NAV_L=100
NAV_ALPHA=1.2
NAV_THREADS=16
# NAV_SAMPLES_PER_PAGE: sample this many vectors from every page in priority order
#   (top first, then last, second-to-last, ...). Default 1 gives every page a nav node.
NAV_SAMPLES_PER_PAGE=1
NAV_SAMPLED_PAGES=0   # 0 = use NAV_SAMPLES_PER_PAGE over all pages

# ============================================================

run_step() {
    local name="$1" log="$2"; shift 2
    echo ""; echo "=== STEP: ${name} ==="; echo "LOG: ${log}"
    "$@" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    [ ${rc} -ne 0 ] && { echo "ERROR: ${name} failed (exit ${rc})"; exit ${rc}; }
    echo "DONE: ${name}"
}

# ---- Pre-flight ----
echo "=== Pre-flight check ==="
MISSING=0
check_file() { [ -f "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }
check_exe()  { [ -x "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }
check_exe "${FVECS_TO_BIN}"; check_exe "${IVECS_TO_BIN}"
check_exe "${BUILD_VAMANA}"; check_exe "${GEN_PAGE}"; check_exe "${BUILD_NAV}"
check_file "${BASE_FVECS}"; check_file "${QUERY_FVECS}"; check_file "${GT_IVECS}"
[ "${MISSING}" -gt 0 ] && { echo "ERROR: ${MISSING} item(s) missing."; exit 1; }
echo "All checks passed."

# ---- Step 0: Convert to bin format ----
if [ ! -f "${BASE_BIN}" ]; then
    run_step "fvecs_to_bin (base)" "${LOG_DIR}/step0a_fvecs_base.log" \
        "${FVECS_TO_BIN}" float "${BASE_FVECS}" "${BASE_BIN}"
else
    echo ""; echo "Skipping base conversion (${BASE_BIN} already exists)"
fi

if [ ! -f "${QUERY_BIN}" ]; then
    run_step "fvecs_to_bin (query)" "${LOG_DIR}/step0b_fvecs_query.log" \
        "${FVECS_TO_BIN}" float "${QUERY_FVECS}" "${QUERY_BIN}"
else
    echo ""; echo "Skipping query conversion (${QUERY_BIN} already exists)"
fi

if [ ! -f "${GT_BIN}" ]; then
    run_step "ivecs_to_bin (groundtruth)" "${LOG_DIR}/step0c_ivecs_gt.log" \
        "${IVECS_TO_BIN}" "${GT_IVECS}" "${GT_BIN}"
else
    echo ""; echo "Skipping GT conversion (${GT_BIN} already exists)"
fi

# ---- Step 1: Build Vamana disk index ----
run_step "build_vamana_disk_index" "${LOG_DIR}/step1_vamana.log" \
    "${BUILD_VAMANA}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_BIN}" \
    --index_path_prefix "${VAMANA_PREFIX}" \
    -R "${VAMANA_R}" \
    -L "${VAMANA_L}" \
    -B "${VAMANA_B}" \
    -M "${VAMANA_M}"

# ---- Step 2: Generate PageANN page graph ----
# --enable_spare_fill is true by default; pass --enable_spare_fill false to skip filling.
run_step "generate_page_graph" "${LOG_DIR}/step2_generate_page.log" \
    "${GEN_PAGE}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_BIN}" \
    --vamana_index_path_prefix "${VAMANA_PREFIX}" \
    --R "${VAMANA_R}" \
    --num_PQ_chunks "${PQ_CHUNKS}" \
    --mem_budget_in_GB "${VAMANA_B}" \
    --full_ooc false \
    --min_degree_per_node "${MIN_DEGREE}"

# ---- Locate PageANN index ----
VAMANA_BASENAME="$(basename "${VAMANA_PREFIX}")"
PAGEANN_INDEX="$(ls "${INDEX_DIR}/${VAMANA_BASENAME}"_PGD*_PageANN.index 2>/dev/null | head -1)"
[ -z "${PAGEANN_INDEX}" ] && { echo "ERROR: PageANN index not found in ${INDEX_DIR}/"; exit 1; }
PAGEANN_PREFIX="${PAGEANN_INDEX%.index}"
echo "PageANN prefix: ${PAGEANN_PREFIX}"

# ---- Step 3: Build in-memory nav graph ----
run_step "build_pageann_nav_graph" "${LOG_DIR}/step3_nav_graph.log" \
    "${BUILD_NAV}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --index_file "${PAGEANN_INDEX}" \
    --output_prefix "${PAGEANN_PREFIX}" \
    --samples_per_page  "${NAV_SAMPLES_PER_PAGE}" \
    --num_sampled_pages "${NAV_SAMPLED_PAGES}" \
    -R "${NAV_R}" \
    -L "${NAV_L}" \
    --alpha "${NAV_ALPHA}" \
    -T "${NAV_THREADS}"

echo ""
echo "###################################################"
echo "# Build complete."
echo "#"
echo "# Final index prefix : ${PAGEANN_PREFIX}"
echo "# Ground truth file  : ${GT_BIN}"
echo "# Query file         : ${QUERY_BIN}"
echo "# Logs               : ${LOG_DIR}/"
echo "#"
echo "# Use these in search_pageann_sift1m.sh:"
echo "#   INDEX_PREFIX=\"${PAGEANN_PREFIX}\""
echo "###################################################"
