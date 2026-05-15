# similar-images

CLI tool to scan a folder, compare images, classify similarity, and generate an HTML report.

## Setup

```bash
uv venv
uv sync
```

## Build executable (PyInstaller)

Build a standalone executable:

```bash
uv run python build_executable.py
```

Output:

- Windows: `dist/similar-images.exe`
- Linux/macOS: `dist/similar-images`

Run executable examples:

```bash
# Windows
./dist/similar-images.exe scan "C:/path/to/images" --output report.html

# Linux/macOS
./dist/similar-images scan "/path/to/images" --output report.html
```

Notes:

- Build on the same OS you plan to run on (no cross-compiling in this setup).
- Re-run `uv sync` before building to ensure PyInstaller and dependencies are installed.

## Usage

```bash
uv run similar-images scan "C:/path/to/images" --output report.html
```

Scan multiple folders in one run:

```bash
uv run similar-images scan "C:/photos/2024" "D:/archive/screenshots" "E:/misc" --output report.html
```

Options:

- `--recursive/--no-recursive` scan nested folders (default recursive)
- `--similar-threshold` score threshold for `similar` (default `0.82`)
- `--duplicate-threshold` score threshold for `duplicate` (default `0.96`)
- `--histogram-weight` weight for histogram feature (default `0.4`)
- `--phash-weight` weight for pHash feature (default `0.2`)
- `--hog-weight` weight for HOG feature (default `0.4`)
- `--output`, `-o` output report path (default `report.html`)

Notes:

- You can pass one or more folder paths after `scan`.
- Images from all provided folders are combined into a single comparison set.
- Exact duplicate file paths are deduplicated automatically before comparison.
- Weights are auto-normalized internally, so they do not need to add up to `1.0`.
- To disable an extraction method, set its weight to `0`.
- At least one extraction weight must be greater than `0`.

Examples for enabling/disabling extraction methods:

```bash
# Use histogram + pHash, disable HOG
uv run similar-images scan "C:/images" --hog-weight 0

# Use only pHash
uv run similar-images scan "C:/images" --histogram-weight 0 --hog-weight 0 --phash-weight 1

# Strongly favor HOG
uv run similar-images scan "C:/images" --histogram-weight 0.1 --phash-weight 0.1 --hog-weight 0.8
```

## Output report

The HTML report includes:

- scanned folder path
- generated timestamp
- threshold values
- summary counts
- table of pair comparisons with:
  - left image name and path
  - right image name and path
  - score
  - classifier (`duplicate`, `similar`, `different`)

## Similarity approach

Per image, the tool extracts:

- normalized color histogram
- DCT-based pHash
- HOG descriptor

Final score uses weighted blend (defaults):

- histogram: 40%
- pHash: 20%
- HOG: 40%

### How similarity is calculated (step by step)

For each image, the tool computes three independent feature vectors and then combines their pairwise similarity into one final score in `[0.0, 1.0]`.

1. **Color histogram similarity (40%)**
   - Build a 3D RGB histogram with `8x8x8` bins (`512` bins total).
   - Normalize it with OpenCV normalization.
   - Compare two histograms with `cv2.compareHist(..., HISTCMP_CORREL)`.
   - Negative correlation values are clamped to `0`.

2. **Perceptual hash similarity (20%)**
   - Convert image to grayscale and resize/pad to `128x128`.
   - Run DCT (`cv2.dct`) and take low-frequency `8x8` block.
   - Build a binary pHash by thresholding coefficients against the block median.
   - Compare two hashes using normalized Hamming distance:
     - `score_phash = 1 - (different_bits / total_bits)`

3. **HOG similarity (40%)**
   - Compute HOG descriptor on the `128x128` normalized grayscale image.
   - Compare two HOG vectors with cosine similarity.
   - Negative cosine values are clamped to `0`.

4. **Final blended score**
   - `score = (w_hist * hist + w_phash * phash + w_hog * hog) / (w_hist + w_phash + w_hog)`
   - Weights are configurable via CLI options (`--histogram-weight`, `--phash-weight`, `--hog-weight`).
   - Score is clamped to `[0, 1]`.

### Classification

Given the final score:

- `duplicate` if `score >= duplicate-threshold` (default `0.96`)
- `similar` if `score >= similar-threshold` (default `0.82`)
- `different` otherwise

### Report presentation

- The HTML report shows **thumbnails for `similar` and `duplicate` rows**.
- Score is shown as both numeric value and a compact progress bar.
- Clicking `similar` or `duplicate` in the classifier column opens a side-by-side compare dialog with larger images.
