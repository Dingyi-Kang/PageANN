#!/bin/bash
# PageANN index build pipeline — generic template
#
# Steps:
#   1. build_vamana_disk_index        — build the Vamana vector-level disk index
#   2. generate_page_graph            — page-align the Vamana graph into PageANN disk format
#   3. Move output files              — relocate to PAGEANN_OUTPUT_DIR
#   4. Remap ground truth             — remap Vamana GT to PageANN's reassigned vector IDs
#   5. build_pageann_nav_graph        — build the in-memory nav graph over sampled page vectors
#
# Usage:
#   Edit the CONFIGURE section below, then:
#   bash scripts/build_pageann_index.sh

set -euo pipefail

# ============================================================
# CONFIGURE — fill in these paths and parameters
# ============================================================

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" 2>/dev/null && pwd || echo "$HOME/DynaANN/build")"

# Where to write the Vamana index and the finished PageANN index
VAMANA_OUTPUT_DIR="/mnt/nvme/<dataset>/diskann"
PAGEANN_OUTPUT_DIR="/mnt/nvme/<dataset>/PageANN"

# Dataset
DATA_TYPE="uint8"         # float | int8 | uint8
DIST_FN="l2"              # l2 | cosine | mips
BASE_DATA="/path/to/base.bin"
QUERY="/path/to/query.bin"

# build_vamana_disk_index params
VAMANA_R=25               # max graph degree
VAMANA_L=150              # build-time search list size
VAMANA_B=5.95             # search-time memory budget in GB (0.5× dataset size)
VAMANA_M=90               # build-time memory budget in GB
N_POINTS=100000000        # number of vectors in the dataset
DIM=128                   # vector dimension

# PQ_CHUNKS is derived from B — do NOT set it independently.
# build_vamana_disk_index computes: PQ_CHUNKS = min(dim, floor(B_bytes / N_points))
# generate_page_graph reads the PQ files produced by the Vamana build and MUST
# receive the same PQ_CHUNKS value. Passing a mismatched value causes a build error.
# To target a specific PQ size, adjust B: B_GB = target_chunks × N_points / 1024³
PQ_CHUNKS=$(python3 -c "import math; print(min(${DIM}, math.floor(${VAMANA_B}*1024**3/${N_POINTS})))")

# Derived Vamana prefix (output of Step 1, input to Step 2)
VAMANA_PREFIX="${VAMANA_OUTPUT_DIR}/vamana_R${VAMANA_R}_L${VAMANA_L}_PQ${PQ_CHUNKS}"

# generate_page_graph params
#
# min_degree_per_node: minimum average Vamana edges per vector that the page graph
#   must satisfy. Controls page size: the tool finds the fewest vectors_per_page
#   such that (page_degree / vectors_per_page) ≥ min_degree_per_node.
#   Set equal to VAMANA_R so the page capacity matches the actual graph quality.
MIN_DEGREE=${VAMANA_R}
#
# full_ooc: set true to minimise in-memory data structures during build.
#   Use when build-time RAM is tight. Typical: false.
FULL_OOC=false
#
# enable_spare_fill: fill spare neighbor slots in each page using beam search.
#   Default is true (recommended). Set false to skip filling and accept pages
#   with fewer than the maximum number of neighbors.
ENABLE_SPARE_FILL=true

# Ground truth remapping
#
# generate_page_graph reassigns vector IDs when grouping vectors into pages,
# making the original Vamana ground truth invalid for the PageANN index.
# Step 4 remaps the GT using the id_map_file produced by generate_page_graph.
#
# GT_K: number of nearest neighbors in the ground truth file (typically 100)
GT_K=100
ORIG_GT="${VAMANA_PREFIX}_gt.bin"   # ground truth produced by build_vamana_disk_index

# build_pageann_nav_graph params
NAV_R=23
NAV_L=100
NAV_ALPHA=1.2
NAV_THREADS=16
#
# NAV_SAMPLES_PER_PAGE: number of vectors to sample from every page.
#   Sampling order: top vector (index 0) first, then last, second-to-last, ...
#   Default 1 guarantees every page has a nav graph representative.
#   Increase to 2+ for a denser nav graph at the cost of more memory.
#
# NAV_SAMPLED_PAGES: when set > 0, switches to page subsampling mode — sample
#   exactly this many evenly distributed pages (1 vector each). Use when memory
#   budget cannot fit one nav node per page.
NAV_SAMPLES_PER_PAGE=1
NAV_SAMPLED_PAGES=0   # 0 = use NAV_SAMPLES_PER_PAGE over all pages

# ============================================================
# (nothing to edit below this line)
# ============================================================

mkdir -p "${VAMANA_OUTPUT_DIR}" "${PAGEANN_OUTPUT_DIR}"
LOG_DIR="${PAGEANN_OUTPUT_DIR}/build_logs"
mkdir -p "${LOG_DIR}"

BUILD_VAMANA="${BUILD_DIR}/apps/build_vamana_disk_index"
GEN_PAGE="${BUILD_DIR}/apps/generate_page_graph"
COMPUTE_GT="${BUILD_DIR}/apps/utils/compute_groundtruth"
BUILD_NAV="${BUILD_DIR}/apps/utils/build_pageann_nav_graph"

# ---- pre-flight ----
echo "=== Pre-flight check ==="
MISSING=0
check_file() { [ -f "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }
check_exe()  { [ -x "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }

echo "Binaries:"
check_exe "${BUILD_VAMANA}"; check_exe "${GEN_PAGE}"; check_exe "${COMPUTE_GT}"; check_exe "${BUILD_NAV}"
echo "Dataset:"
check_file "${BASE_DATA}"; check_file "${QUERY}"

[ "${MISSING}" -gt 0 ] && { echo ""; echo "ERROR: ${MISSING} file(s) missing. Aborting."; exit 1; }
echo "All checks passed. Starting pipeline."

# ---- helper ----
run_step() {
    local name="$1" log="$2"; shift 2
    echo ""; echo "========================================"; echo "STEP: ${name}"; echo "LOG:  ${log}"; echo "========================================"
    "$@" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    [ ${rc} -ne 0 ] && { echo "ERROR: ${name} failed (exit ${rc})" | tee -a "${log}"; exit ${rc}; }
    echo "DONE: ${name}" | tee -a "${log}"
}

echo "Build dir:         ${BUILD_DIR}"
echo "Vamana out dir:    ${VAMANA_OUTPUT_DIR}"
echo "PageANN out dir:   ${PAGEANN_OUTPUT_DIR}"

# ---- Step 1: build Vamana disk index ----
run_step "build_vamana_disk_index" "${LOG_DIR}/step1_build_vamana.log" \
    "${BUILD_VAMANA}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_DATA}" \
    --index_path_prefix "${VAMANA_PREFIX}" \
    -R "${VAMANA_R}" \
    -L "${VAMANA_L}" \
    -B "${VAMANA_B}" \
    -M "${VAMANA_M}"

# ---- Step 2: generate PageANN page graph ----
run_step "generate_page_graph" "${LOG_DIR}/step2_generate_page_graph.log" \
    "${GEN_PAGE}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_DATA}" \
    --vamana_index_path_prefix "${VAMANA_PREFIX}" \
    --R "${VAMANA_R}" \
    --min_degree_per_node "${MIN_DEGREE}" \
    --num_PQ_chunks "${PQ_CHUNKS}" \
    --mem_budget_in_GB "${VAMANA_B}" \
    --full_ooc "${FULL_OOC}" \
    --enable_spare_fill "${ENABLE_SPARE_FILL}"

# ---- Step 3: move generated files to output dir ----
echo ""; echo "========================================"; echo "STEP: Move PageANN files to output dir"; echo "========================================"
VAMANA_BASENAME="$(basename "${VAMANA_PREFIX}")"
VAMANA_DIR="$(dirname "${VAMANA_PREFIX}")"

shopt -s nullglob
FILES_TO_MOVE=("${VAMANA_DIR}/${VAMANA_BASENAME}"_PGD*_PageANN*)
shopt -u nullglob

[ ${#FILES_TO_MOVE[@]} -eq 0 ] && { echo "ERROR: No PageANN output files found at ${VAMANA_DIR}/${VAMANA_BASENAME}_PGD*_PageANN*"; exit 1; }
echo "Moving ${#FILES_TO_MOVE[@]} file(s) to ${PAGEANN_OUTPUT_DIR}/"
for f in "${FILES_TO_MOVE[@]}"; do echo "  mv $(basename "$f")"; mv "$f" "${PAGEANN_OUTPUT_DIR}/"; done
echo "DONE: Move PageANN files"

# Discover relocated index
PAGEANN_INDEX="$(ls "${PAGEANN_OUTPUT_DIR}/${VAMANA_BASENAME}"_PGD*_PageANN.index 2>/dev/null | head -1)"
[ -z "${PAGEANN_INDEX}" ] && { echo "ERROR: Could not find PageANN index in ${PAGEANN_OUTPUT_DIR}"; exit 1; }
PAGEANN_PREFIX="${PAGEANN_INDEX%.index}"
echo "Detected PageANN prefix: ${PAGEANN_PREFIX}"

# ---- Step 4: remap ground truth ----
# generate_page_graph reassigns vector IDs, so the Vamana GT must be remapped.
PAGEANN_GT="${PAGEANN_PREFIX}_gt.bin"
ID_MAP="${PAGEANN_PREFIX}_new_to_old_ids_map.bin"

[ -f "${ID_MAP}" ] || { echo "ERROR: ID map not found: ${ID_MAP}"; exit 1; }

run_step "remap_groundtruth" "${LOG_DIR}/step4_remap_gt.log" \
    "${COMPUTE_GT}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --base_file "${BASE_DATA}" \
    --query_file "${QUERY}" \
    --gt_file "${PAGEANN_GT}" \
    --K "${GT_K}" \
    --id_map_file "${ID_MAP}" \
    --vamana_gt_file "${ORIG_GT}" \
    --skip_knn

# ---- Step 5: build nav graph ----
run_step "build_pageann_nav_graph" "${LOG_DIR}/step5_build_nav_graph.log" \
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
echo "# Ground truth file  : ${PAGEANN_GT}"
echo "# Logs               : ${LOG_DIR}/"
echo "#"
echo "# Use these values in search_pageann_index.sh:"
echo "#   INDEX_PREFIX=\"${PAGEANN_PREFIX}\""
echo "#   GT_FILE=\"${PAGEANN_GT}\""
echo "###################################################"
