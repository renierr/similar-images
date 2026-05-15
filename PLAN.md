# Similar Images Tool Plan

## Goal

Build a Python CLI tool that scans an image folder, compares images, classifies pair similarity, and writes an HTML report with file path, file name, score, and classifier.

## Scope (MVP)

- Read image files from a folder (recursive by default).
- Build features per image with OpenCV-inspired approach from `heresphere-server`:
  - color histogram
  - DCT pHash
  - HOG descriptor
- Compute weighted score:
  - histogram 40%
  - pHash 20%
  - HOG 40%
- Classify each pair as:
  - `duplicate`
  - `similar`
  - `different`
- Generate HTML report containing:
  - scanned folder metadata
  - summary counts
  - table of all compared pairs with name, path, score, classifier

## Code structure

- `src/similar_images/io.py`: folder scanning and supported extension filter
- `src/similar_images/features.py`: image feature extraction
- `src/similar_images/similarity.py`: score + classifier logic
- `src/similar_images/classifier.py`: all-pairs comparison engine
- `src/similar_images/report.py`: HTML report generation
- `src/similar_images/cli.py`: Typer CLI

## Notes

- Python version intentionally allows latest 3.x (`>=3.10,<4`), not pinned to one minor version.
- Initial implementation is O(n^2) for pair comparisons; acceptable for moderate folder sizes.
- Next refinement can add candidate pruning/indexing for large datasets.

## Refinement backlog

- Add optional thumbnail previews in the report.
- Add CSV/JSON exports alongside HTML.
- Add grouping view (connected components of similar/duplicate links).
- Add tests for scoring and classification boundaries.
- Add parallel feature extraction for large folders.
