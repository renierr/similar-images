# similar-images

Tool to scan a folder, compare images, classify similarity, and generate an HTML report. Now with a modern Graphical User Interface (GUI)!

## Setup

This project uses `uv` for package management.

```bash
uv venv
uv sync
```

## GUI Usage (Recommended)

To launch the modern graphical interface:

```bash
uv run similar-images gui
# or
uv run similar-images-gui
```

The GUI allows you to easily:
- Select multiple folders to scan.
- Adjust similarity thresholds and weights via sliders.
- Monitor scan progress.
- Open the generated report automatically.

## CLI Usage

```bash
uv run similar-images scan "C:/path/to/images" --output report.html
```

Scan multiple folders in one run:

```bash
uv run similar-images scan "C:/photos/2024" "D:/archive/screenshots" --output report.html
```

### Options

- `--recursive/--no-recursive` scan nested folders (default recursive)
- `--similar-threshold` score threshold for `similar` (default `0.82`)
- `--duplicate-threshold` score threshold for `duplicate` (default `0.96`)
- `--histogram-weight` weight for histogram feature (default `0.3`)
- `--phash-weight` weight for pHash feature (default `0.2`)
- `--dhash-weight` weight for dHash feature (default `0.2`)
- `--hog-weight` weight for HOG feature (default `0.3`)
- `--orb-weight` weight for ORB keypoint match feature (default `0.0`)
- `--ssim-weight` weight for SSIM grayscale structure similarity (default `0.0`)
- `--edge-weight` weight for Canny edge-structure similarity (default `0.0`)
- `--report-min-score` minimum score to include rows in report (default `0.3`)
- `--report-max-rows` maximum report rows; `0` means unlimited (default `0`)
- `--reset-weights` set all similarity weights to 0.0 before applying other options
- `--output`, `-o` output report path (default `report.html`)

### Resetting Weights

If you want to use only a specific similarity method, use `--reset-weights`:

```bash
# Use ONLY dHash
uv run similar-images scan "C:/images" --reset-weights --dhash-weight 1.0
```

## Build executable (PyInstaller)

Build a standalone executable (GUI and CLI included):

```bash
uv run python build_executable.py
```

Output:
- Windows: `dist/similar-images.exe`
- Linux/macOS: `dist/similar-images`

## Similarity Approach

The tool uses a multi-resolution approach to remain robust against different qualities and resolutions:

1. **Stretched 32x32 Processing:** For perceptual hashes (**pHash** and **dHash**), images are stretched to a small square. This removes noise and ignores aspect ratio differences.
2. **HSV Histograms:** Uses Hue-Saturation-Value color space, which is much more robust to lighting and quality changes than raw RGB.
3. **Structural 128x128 Processing:** **HOG**, **SSIM**, and **Edge** detection use a higher resolution to capture structural details.

### Final Blended Score (Defaults)
- **Histogram:** 30% (HSV)
- **pHash:** 20% (32x32 stretched)
- **dHash:** 20% (32x32 stretched)
- **HOG:** 30% (128x128 stretched)
- **Others:** Opt-in (ORB, SSIM, Edge)

## Classification

- `duplicate` if `score >= 0.96`
- `similar` if `score >= 0.82`
- `different` otherwise (pairs below `0.2` are omitted by default)
