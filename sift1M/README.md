# SIFT1M Dataset

SIFT1M is a standard ANN benchmark dataset of 1 million 128-dimensional float vectors extracted from images using the SIFT descriptor.

**Download**: [https://huggingface.co/datasets/qbo-odp/sift1m/tree/main](https://huggingface.co/datasets/qbo-odp/sift1m/tree/main)

**Original source**: [http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)

**Citation**:
> Jégou H, Douze M, Schmid C. Improving bag-of-features for large scale image search. International Journal of Computer Vision, 2010, 87(3): 316–336.

---

## Download

Download the three files from HuggingFace and place them in this directory (`sift1M/`):

```bash
cd ~/DynaANN/sift1M

wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs
wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs
wget https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs
```

---

## Files

| File | Size | Description |
|---|---|---|
| `sift_base.fvecs` | 493 MB | 1,000,000 base vectors (128-dim float) |
| `sift_query.fvecs` | 5 MB | 10,000 query vectors (128-dim float) |
| `sift_groundtruth.ivecs` | 3.9 MB | 100 ground truth nearest neighbors per query |

**Format**: `.fvecs` and `.ivecs` are the standard TexMex binary formats. Each vector is stored as a 4-byte integer (dimension) followed by the vector data. LAANN requires plain `.bin` format — see conversion below.

---

## Format Conversion

LAANN uses a simple binary format: an 8-byte header (`num_points`, `num_dims` as uint32) followed by raw vector data. The `.fvecs` and `.ivecs` files must be converted before building — **`build_laann_sift1m.sh` handles this automatically** and skips the conversion if the `.bin` files already exist.

---

## Building and Searching with LAANN

Use the provided scripts (run from the DynaANN root directory):

```bash
# Build the LAANN index
bash sift1M/build_laann_sift1m.sh

# Search (edit INDEX_PREFIX inside the script after the build completes)
bash sift1M/search_laann_sift1m.sh
```

See [`build_laann_sift1m.sh`](build_laann_sift1m.sh) and [`search_laann_sift1m.sh`](search_laann_sift1m.sh) for full parameter details.

---

## Dataset Summary

| Property | Value |
|---|---|
| Vectors | 1,000,000 |
| Dimensions | 128 |
| Data type | float32 |
| Queries | 10,000 |
| Ground truth K | 100 |
| Distance metric | L2 |
