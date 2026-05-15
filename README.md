# similar-images

CLI tool to scan a folder, compare images, classify similarity, and generate an HTML report.

## Setup

```bash
uv venv
uv sync
```

## Usage

```bash
uv run similar-images scan "C:/path/to/images" --output report.html
```

Options:

- `--recursive/--no-recursive` scan nested folders (default recursive)
- `--similar-threshold` score threshold for `similar` (default `0.82`)
- `--duplicate-threshold` score threshold for `duplicate` (default `0.96`)
- `--output`, `-o` output report path (default `report.html`)

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

The comparison logic is inspired by `../heresphere-server/src/similar.py` and adapted for still images.

Per image, the tool extracts:

- normalized color histogram
- DCT-based pHash
- HOG descriptor

Final score uses weighted blend:

- histogram: 40%
- pHash: 20%
- HOG: 40%
