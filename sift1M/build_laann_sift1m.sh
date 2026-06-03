#!/bin/bash
# Build LAANN index for SIFT1M
#
# Steps:
#   0. Convert fvecs/ivecs → bin format
#   1. Build Vamana disk index
#   2. Generate LAANN page graph
#   3. Move LAANN files to output dir
#   4. Build lightweight in-memory nav graph
#   5. Reorder pages by frequency + remap ground truth (fsort)
#
# Run from the DynaANN root:
#   bash sift1M/build_laann_sift1m.sh

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
GEN_LAANN="${BUILD_DIR}/apps/generate_laann_graph"
BUILD_NAV="${BUILD_DIR}/apps/utils/build_laann_nav_graph"
REORDER="${BUILD_DIR}/apps/utils/reorder_pages_by_frequency"

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
# R should match the page-level degree per vector so Vamana fills the page at full
# quality without heavy reliance on the fill phase.
# Page capacity = 6 vectors/page, page degree = 252 (42×6) → R = 252/6 = 42.
# With R=25 the fill phase must add 252-150=102 lower-quality edges per page.
VAMANA_R=42
VAMANA_L=120   # keep L > R
VAMANA_B=0.5     # search-time memory budget in GB (≈ 0.5× dataset size)
VAMANA_M=4       # build-time memory budget in GB
N_POINTS=1000000
DIM=128

# PQ_CHUNKS: build_vamana_disk_index derives num_pq_chunks as:
#   PQ_CHUNKS = min(dim, floor(B_bytes / N_points))
# For SIFT1M: min(128, floor(0.5×1024³ / 1M)) = min(128, 536) = 128
# generate_laann_graph MUST use the same value — they share the PQ files.
# To target ~20% of original byte size (512 bytes/vector): ~102 chunks → set B≈0.10 GB.
# Here we use B=0.5 → PQ_CHUNKS=128 (25% of 512 bytes), which is fine for demo.
PQ_CHUNKS=$(python3 -c "import math; print(min(${DIM}, math.floor(${VAMANA_B}*1024**3/${N_POINTS})))")
echo "Derived PQ_CHUNKS=${PQ_CHUNKS} from B=${VAMANA_B} GB, N=${N_POINTS}, dim=${DIM}"
VAMANA_PREFIX="${INDEX_DIR}/vamana_sift1M_R${VAMANA_R}_L${VAMANA_L}_PQ${PQ_CHUNKS}"

# ---- LAANN page graph params ----
# min_degree_per_node: minimum average Vamana edges per vector that the page graph
#   must satisfy. The tool finds the smallest vectors_per_page such that
#   (page_degree / vectors_per_page) ≥ min_degree_per_node.
#   Set equal to VAMANA_R so the capacity calculation matches the actual graph quality:
#   with R=42 and capacity=6, page_degree=252=42×6, perfectly filled by Vamana edges.
MIN_DEGREE=${VAMANA_R}
# grouping_L: search list size during the page-grouping phase. Vectors are
#   grouped by running a greedy search with beam width L; higher L produces
#   spatially tighter pages (better locality) at the cost of a slower build.
GROUPING_L=200
# fill_L: search list size during the neighbor-filling phase. After vectors are
#   grouped into pages, the page-level graph edges are filled using a beam search
#   of this size. Higher fill_L → better edge quality, slower build.
FILL_L=100

# ---- Nav graph params ----
NAV_R=24
NAV_L=100
NAV_ALPHA=1.2
NAV_THREADS=16
NAV_SAMPLES=1

# ---- Fsort params ----
SAMPLE_RATIO=0.05    # larger ratio for small dataset
ORIG_PQ_COMPRESSED="${VAMANA_PREFIX}_pq_compressed.bin"
# ORIG_GT: use the converted sift_groundtruth.bin (from Step 0 ivecs conversion).
# Unlike large-scale datasets where build_vamana_disk_index produces a _gt.bin,
# for SIFT1M we use the dataset's own ground truth directly.
ORIG_GT="${GT_BIN}"

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
check_exe "${BUILD_VAMANA}"; check_exe "${GEN_LAANN}"
check_exe "${BUILD_NAV}";    check_exe "${REORDER}"
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

# ---- Step 2: Generate LAANN page graph ----
run_step "generate_laann_graph" "${LOG_DIR}/step2_generate_laann.log" \
    "${GEN_LAANN}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --data_path "${BASE_BIN}" \
    --vamana_index_path_prefix "${VAMANA_PREFIX}" \
    --R "${VAMANA_R}" \
    --min_degree_per_node "${MIN_DEGREE}" \
    --num_PQ_chunks "${PQ_CHUNKS}" \
    --L "${GROUPING_L}" \
    --fill_L "${FILL_L}"

# ---- Step 3: Move LAANN files to index dir ----
echo ""; echo "=== Moving LAANN files to ${INDEX_DIR}/ ==="
VAMANA_BASENAME="$(basename "${VAMANA_PREFIX}")"
shopt -s nullglob
FILES=("${INDEX_DIR}/${VAMANA_BASENAME}"_MGD*_LAANN*)
shopt -u nullglob
[ ${#FILES[@]} -eq 0 ] && { echo "ERROR: No LAANN output files found."; exit 1; }
echo "Found ${#FILES[@]} file(s) — already in ${INDEX_DIR}/, no move needed."

LAANN_INDEX="$(ls "${INDEX_DIR}/${VAMANA_BASENAME}"_MGD*_LAANN.index 2>/dev/null | head -1)"
[ -z "${LAANN_INDEX}" ] && { echo "ERROR: LAANN index not found."; exit 1; }
LAANN_PREFIX="${LAANN_INDEX%.index}"
echo "LAANN prefix: ${LAANN_PREFIX}"

# ---- Step 4: Build nav graph ----
run_step "build_laann_nav_graph" "${LOG_DIR}/step4_nav_graph.log" \
    "${BUILD_NAV}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --laann_disk_index_file "${LAANN_INDEX}" \
    --samples_per_page "${NAV_SAMPLES}" \
    -R "${NAV_R}" \
    -L "${NAV_L}" \
    --alpha "${NAV_ALPHA}" \
    -T "${NAV_THREADS}"

# ---- Step 5: Reorder pages by frequency + remap GT ----
run_step "reorder_pages_by_frequency" "${LOG_DIR}/step5_reorder.log" \
    "${REORDER}" \
    --data_type "${DATA_TYPE}" \
    --dist_fn   "${DIST_FN}" \
    --index_path_prefix "${LAANN_PREFIX}" \
    --data_bin "${BASE_BIN}" \
    --sample_ratio "${SAMPLE_RATIO}" \
    --orig_pq_compressed_file "${ORIG_PQ_COMPRESSED}" \
    --orig_gt_file "${ORIG_GT}"

FINAL_PREFIX="${LAANN_PREFIX}_fsort"
FINAL_GT="${LAANN_PREFIX}_fsort_gt.bin"

echo ""
echo "###################################################"
echo "# Build complete."
echo "#"
echo "# Final index prefix : ${FINAL_PREFIX}"
echo "# Ground truth file  : ${FINAL_GT}"
echo "# Query file         : ${QUERY_BIN}"
echo "# Logs               : ${LOG_DIR}/"
echo "#"
echo "# Use these in search_laann_sift1m.sh:"
echo "#   INDEX_PREFIX=\"${FINAL_PREFIX}\""
echo "#   GT_FILE=\"${FINAL_GT}\""
echo "###################################################"
