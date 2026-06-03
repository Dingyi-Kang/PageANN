#!/bin/bash
# LAANN index build pipeline — generic template
#
# Steps:
#   1. build_vamana_disk_index        — build the Vamana vector-level disk index
#   2. generate_laann_graph           — page-align the Vamana graph into LAANN disk format
#   3. Move output files              — relocate to LAANN_OUTPUT_DIR
#   4. build_laann_nav_graph          — build the lightweight in-memory nav graph over page centroids
#   5. reorder_pages_by_frequency     — reorder pages by frequency + remap ground truth (skip at your own risk)
#
# Usage:
#   Edit the CONFIGURE section below, then:
#   bash scripts/build_laann_index.sh

set -euo pipefail

# ============================================================
# CONFIGURE — fill in these paths and parameters
# ============================================================

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" 2>/dev/null && pwd || echo "$HOME/DynaANN/build")"

# Where to write the Vamana index and the finished LAANN index
VAMANA_OUTPUT_DIR="/mnt/nvme/<dataset>/diskann"
LAANN_OUTPUT_DIR="/mnt/nvme/<dataset>/LAANN"

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
# generate_laann_graph reads the PQ files produced by the Vamana build and MUST
# receive the same PQ_CHUNKS value. Passing a mismatched value causes a build error.
# To target a specific PQ size, adjust B: B_GB = target_chunks × N_points / 1024³
PQ_CHUNKS=$(python3 -c "import math; print(min(${DIM}, math.floor(${VAMANA_B}*1024**3/${N_POINTS})))")

# Derived Vamana prefix (output of Step 1, input to Step 2)
VAMANA_PREFIX="${VAMANA_OUTPUT_DIR}/vamana_R${VAMANA_R}_L${VAMANA_L}_PQ${PQ_CHUNKS}"

# generate_laann_graph params
#
# min_degree_per_node: minimum average Vamana edges per vector that the page graph
#   must satisfy. Controls page size: the tool finds the fewest vectors_per_page
#   such that (page_degree / vectors_per_page) ≥ min_degree_per_node.
#   Set equal to VAMANA_R: this ensures the page capacity matches the actual graph
#   quality — Vamana provides R edges per vector, which exactly fills the page
#   (page_degree = R × vectors_per_page). If MIN_DEGREE < R, the page can fit more
#   vectors but Vamana must then fill spare neighbor slots with lower-quality edges.
MIN_DEGREE=${VAMANA_R}
#
# grouping_L: search list size during the page-grouping phase. Vectors are
#   grouped by running a greedy beam search; higher L produces spatially tighter
#   pages (better cache locality) at the cost of a slower build.
#   Must be ≥ vectors_per_page. Typical range: 200–400.
GROUPING_L=350
#
# fill_L: search list size during the neighbor-filling phase. After grouping,
#   page-level graph edges are filled using a beam search of this size.
#   Higher fill_L → better edge quality, slower build. Typical range: 100–200.
FILL_L=200

# build_laann_nav_graph params
NAV_R=24
NAV_L=150
NAV_ALPHA=1.2
NAV_THREADS=16
NAV_SAMPLES=1

# Step 4: reorder_pages_by_frequency
#
# generate_laann_graph reassigns vector IDs when grouping vectors into pages,
# making the original Vamana ground truth invalid for the LAANN index.
# reorder_pages_by_frequency applies the compound remapping (page grouping +
# frequency sort) to the GT and writes <prefix>_fsort_gt.bin.
#
# DO_FSORT=true  (recommended) → pages reordered by frequency + GT remapped.
#                  Search must use: --index_path_prefix <prefix>_fsort
#                                   --gt_file <prefix>_fsort_gt.bin
#
# DO_FSORT=false → pages NOT reordered. You must separately remap the GT using:
#                  apps/utils/compute_groundtruth --skip_knn
#                    --id_map_file <prefix>_new_to_old_ids_map.bin
#                    --vamana_gt_file <original_gt>
#                    --gt_file <output_gt>
#                  Search must use: --index_path_prefix <prefix>  (no _fsort)
#                                   --gt_file <output_gt from above>
DO_FSORT=true
SAMPLE_RATIO=0.01
ORIG_PQ_COMPRESSED="${VAMANA_PREFIX}_pq_compressed.bin"
ORIG_GT="${VAMANA_PREFIX}_gt.bin"   # ground truth produced by build_vamana_disk_index

# ============================================================
# (nothing to edit below this line)
# ============================================================

mkdir -p "${VAMANA_OUTPUT_DIR}" "${LAANN_OUTPUT_DIR}"
LOG_DIR="${LAANN_OUTPUT_DIR}/build_logs"
mkdir -p "${LOG_DIR}"

BUILD_VAMANA="${BUILD_DIR}/apps/build_vamana_disk_index"
GEN_LAANN="${BUILD_DIR}/apps/generate_laann_graph"
BUILD_NAV="${BUILD_DIR}/apps/utils/build_laann_nav_graph"
REORDER="${BUILD_DIR}/apps/utils/reorder_pages_by_frequency"

# ---- pre-flight ----
echo "=== Pre-flight check ==="
MISSING=0
check_file() { [ -f "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }
check_exe()  { [ -x "$1" ] && echo "  OK      $1" || { echo "  MISSING $1"; MISSING=$((MISSING+1)); }; }

echo "Binaries:"
check_exe "${BUILD_VAMANA}"; check_exe "${GEN_LAANN}"; check_exe "${BUILD_NAV}"; check_exe "${REORDER}"
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

echo "Build dir:       ${BUILD_DIR}"
echo "Vamana out dir:  ${VAMANA_OUTPUT_DIR}"
echo "LAANN out dir:   ${LAANN_OUTPUT_DIR}"

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

# ---- Step 2: generate LAANN page graph ----
run_step "generate_laann_graph" "${LOG_DIR}/step2_generate_laann_graph.log" \
    "${GEN_LAANN}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_DATA}" \
    --vamana_index_path_prefix "${VAMANA_PREFIX}" \
    --R "${VAMANA_R}" \
    --min_degree_per_node "${MIN_DEGREE}" \
    --num_PQ_chunks "${PQ_CHUNKS}" \
    --L "${GROUPING_L}" \
    --fill_L "${FILL_L}"

# ---- Step 2b: move generated files to output dir ----
echo ""; echo "========================================"; echo "STEP: Move LAANN files to output dir"; echo "========================================"
VAMANA_BASENAME="$(basename "${VAMANA_PREFIX}")"
VAMANA_DIR="$(dirname "${VAMANA_PREFIX}")"

shopt -s nullglob
FILES_TO_MOVE=("${VAMANA_DIR}/${VAMANA_BASENAME}"_MGD*_LAANN*)
shopt -u nullglob

[ ${#FILES_TO_MOVE[@]} -eq 0 ] && { echo "ERROR: No LAANN output files found at ${VAMANA_DIR}/${VAMANA_BASENAME}_MGD*_LAANN*"; exit 1; }
echo "Moving ${#FILES_TO_MOVE[@]} file(s) to ${LAANN_OUTPUT_DIR}/"
for f in "${FILES_TO_MOVE[@]}"; do echo "  mv $(basename "$f")"; mv "$f" "${LAANN_OUTPUT_DIR}/"; done
echo "DONE: Move LAANN files"

# Discover relocated index
LAANN_INDEX="$(ls "${LAANN_OUTPUT_DIR}/${VAMANA_BASENAME}"_MGD*_LAANN.index 2>/dev/null | head -1)"
[ -z "${LAANN_INDEX}" ] && { echo "ERROR: Could not find LAANN index in ${LAANN_OUTPUT_DIR}"; exit 1; }
LAANN_PREFIX="${LAANN_INDEX%.index}"
echo "Detected LAANN prefix: ${LAANN_PREFIX}"

# ---- Step 3: build nav graph ----
run_step "build_laann_nav_graph" "${LOG_DIR}/step3_build_nav_graph.log" \
    "${BUILD_NAV}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --laann_disk_index_file "${LAANN_INDEX}" \
    --samples_per_page "${NAV_SAMPLES}" \
    -R "${NAV_R}" \
    -L "${NAV_L}" \
    --alpha "${NAV_ALPHA}" \
    -T "${NAV_THREADS}"

# ---- Step 4 (optional): reorder pages by frequency (fsort) ----
if [ "${DO_FSORT}" = "true" ]; then
    run_step "reorder_pages_by_frequency" "${LOG_DIR}/step4_reorder_pages.log" \
        "${REORDER}" \
        --data_type "${DATA_TYPE}" \
        --dist_fn   "${DIST_FN}" \
        --index_path_prefix "${LAANN_PREFIX}" \
        --data_bin "${BASE_DATA}" \
        --sample_ratio "${SAMPLE_RATIO}" \
        --orig_pq_compressed_file "${ORIG_PQ_COMPRESSED}" \
        --orig_gt_file "${ORIG_GT}"
    FINAL_PREFIX="${LAANN_PREFIX}_fsort"
    FINAL_GT="${LAANN_PREFIX}_fsort_gt.bin"
else
    echo ""; echo "Skipping reorder_pages_by_frequency (DO_FSORT=false)."
    echo "NOTE: remap the ground truth before searching using:"
    echo "  ./build/apps/utils/compute_groundtruth \\"
    echo "    --data_type ${DATA_TYPE} --dist_fn ${DIST_FN} \\"
    echo "    --base_file ${BASE_DATA} \\"
    echo "    --query_file ${QUERY} \\"
    echo "    --gt_file ${LAANN_PREFIX}_gt.bin \\"
    echo "    --K 100 \\"
    echo "    --id_map_file ${LAANN_PREFIX}_new_to_old_ids_map.bin \\"
    echo "    --vamana_gt_file ${ORIG_GT} \\"
    echo "    --skip_knn"
    FINAL_PREFIX="${LAANN_PREFIX}"
    FINAL_GT="${LAANN_PREFIX}_gt.bin  (run compute_groundtruth above first)"
fi

echo ""
echo "###################################################"
echo "# Build complete."
echo "#"
echo "# Final index prefix : ${FINAL_PREFIX}"
echo "# Ground truth file  : ${FINAL_GT}"
echo "# Logs               : ${LOG_DIR}/"
echo "#"
echo "# Use these values in search_laann_index.sh:"
echo "#   INDEX_PREFIX=\"${FINAL_PREFIX}\""
[ "${DO_FSORT}" = "true" ] && echo "#   GT_FILE=\"${FINAL_GT}\""
echo "###################################################"
