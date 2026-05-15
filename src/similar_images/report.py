from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .models import PairResult
from .similarity import SimilarityWeights


def _row_class(classifier: str) -> str:
    if classifier == "duplicate":
        return "duplicate"
    if classifier == "similar":
        return "similar"
    return "different"


def _summary_counts(results: list[PairResult]) -> tuple[int, int, int]:
    duplicates = sum(1 for r in results if r.classifier == "duplicate")
    similars = sum(1 for r in results if r.classifier == "similar")
    differents = sum(1 for r in results if r.classifier == "different")
    return duplicates, similars, differents


def _compact_path(image_path: Path, scanned_folders: list[Path]) -> str:
    for folder in scanned_folders:
        try:
            return str(image_path.relative_to(folder))
        except ValueError:
            continue
    return str(image_path)


def _thumbnail_html(image_id: int, classifier: str) -> str:
    if classifier not in {"similar", "duplicate"}:
        return "<span class='muted'>-</span>"
    return f"<img class='thumb js-thumb' data-img-id='{image_id}' alt='thumbnail' loading='lazy' />"


def _score_gauge_html(score: float) -> str:
    pct = max(0.0, min(100.0, score * 100.0))
    return "".join(
        [
            "<div class='score-wrap'>",
            f"<div class='score-number'>{score:.4f}</div>",
            f"<div class='gauge'><span class='fill' style='width:{pct:.2f}%'></span></div>",
            "</div>",
        ]
    )


def _image_cell_html(
    image_id: int,
    image_name: str,
    compact_path: str,
    classifier: str,
) -> str:
    return "".join(
        [
            "<div class='image-cell'>",
            _thumbnail_html(image_id, classifier),
            "<div class='image-meta'>",
            f"<div class='image-name'>{escape(image_name)}</div>",
            f"<div class='image-path'>{escape(compact_path)}</div>",
            "</div>",
            "</div>",
        ]
    )


def _classifier_cell_html(item: PairResult, left_id: int, right_id: int) -> str:
    if item.classifier in {"similar", "duplicate"}:
        classifier = escape(item.classifier)
        return (
            "<button class='compare-btn'"
            f" data-left-id='{left_id}'"
            f" data-right-id='{right_id}'"
            f" data-score='{item.score:.4f}'"
            f" data-classifier='{classifier}'"
            f">{classifier}</button>"
        )
    return "<span class='classifier muted'>different</span>"


def build_html_report(
    scanned_folders: list[Path],
    output_path: Path,
    results: list[PairResult],
    loaded_count: int,
    skipped_count: int,
    similar_threshold: float,
    duplicate_threshold: float,
    weights: SimilarityWeights,
) -> None:
    duplicates, similars, differents = _summary_counts(results)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    scanned_folders_html = "".join(
        [f"<li>{escape(str(folder))}</li>" for folder in scanned_folders]
    )

    image_to_id: dict[Path, int] = {}
    image_registry: list[dict[str, str]] = []

    def image_id_for(path: Path, name: str) -> int:
        existing = image_to_id.get(path)
        if existing is not None:
            return existing
        new_id = len(image_registry)
        image_to_id[path] = new_id
        image_registry.append(
            {
                "name": name,
                "path": str(path),
                "compact_path": _compact_path(path, scanned_folders),
                "uri": path.as_uri(),
            }
        )
        return new_id

    rows = []
    for item in results:
        left_id = image_id_for(item.left.path, item.left.name)
        right_id = image_id_for(item.right.path, item.right.name)

        rows.append(
            "".join(
                [
                    f"<tr class='{_row_class(item.classifier)}'>",
                    f"<td>{_image_cell_html(left_id, item.left.name, image_registry[left_id]['compact_path'], item.classifier)}</td>",
                    f"<td>{_image_cell_html(right_id, item.right.name, image_registry[right_id]['compact_path'], item.classifier)}</td>",
                    f"<td>{_score_gauge_html(item.score)}</td>",
                    f"<td class='classifier'>{_classifier_cell_html(item, left_id, right_id)}</td>",
                    "</tr>",
                ]
            )
        )

    table_body = "\n".join(rows) if rows else "<tr><td colspan='4'>No comparable image pairs found.</td></tr>"
    image_registry_json = json.dumps(image_registry)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Similar Images Report</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --accent: #0f766e;
      --dup: #dcfce7;
      --sim: #fef9c3;
      --diff: #fee2e2;
      --border: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Segoe UI', Tahoma, sans-serif; color: var(--ink); background: radial-gradient(circle at top right, #dbeafe, var(--bg)); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04); }}
    h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
    .meta {{ color: var(--muted); font-size: 0.95rem; }}
    .scan-list {{ margin: 6px 0 8px 20px; padding: 0; color: #334155; }}
    .scan-list li {{ margin: 2px 0; word-break: break-all; font-family: 'Consolas', 'Courier New', monospace; font-size: 0.86rem; }}
    .stats {{ margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }}
    .stat .k {{ color: var(--muted); font-size: 0.85rem; }}
    .stat .v {{ font-size: 1.2rem; font-weight: 700; }}
    .table-wrap {{ margin-top: 18px; overflow: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.92rem; }}
    th {{ background: #f1f5f9; position: sticky; top: 0; z-index: 1; }}
    tr.duplicate td {{ background: var(--dup); }}
    tr.similar td {{ background: var(--sim); }}
    tr.different td {{ background: var(--diff); }}
    .muted {{ color: var(--muted); }}
    .image-cell {{ display: grid; grid-template-columns: 88px 1fr; gap: 10px; align-items: center; min-width: 300px; }}
    .thumb {{ display: block; width: 88px; height: 56px; object-fit: cover; border-radius: 8px; border: 1px solid #cbd5e1; background: #fff; }}
    .image-name {{ font-weight: 600; color: #0f172a; }}
    .image-path {{ margin-top: 2px; color: #334155; font-size: 0.8rem; word-break: break-all; font-family: 'Consolas', 'Courier New', monospace; }}
    .score-wrap {{ width: 120px; }}
    .gauge {{ width: 100%; height: 8px; border-radius: 999px; border: 1px solid #cbd5e1; background: #e2e8f0; overflow: hidden; }}
    .fill {{ display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f97316, #22c55e); }}
    .score-number {{ margin-bottom: 5px; font-variant-numeric: tabular-nums; color: #0f172a; font-weight: 700; }}
    .classifier {{ text-transform: capitalize; font-weight: 600; }}
    .compare-btn {{
      border: 1px solid #94a3b8;
      background: #f8fafc;
      border-radius: 999px;
      padding: 4px 10px;
      cursor: pointer;
      font-weight: 700;
      text-transform: capitalize;
      color: #0f172a;
    }}
    .compare-btn:hover {{ background: #e2e8f0; }}
    .compare-btn:focus {{ outline: 2px solid #0f766e; outline-offset: 2px; }}
    .col-score {{ width: 140px; }}
    .col-classifier {{ width: 110px; }}
    dialog.compare-dialog {{ border: 1px solid var(--border); border-radius: 14px; max-width: 1100px; width: 94vw; padding: 0; }}
    dialog.compare-dialog::backdrop {{ background: rgba(2, 6, 23, 0.55); }}
    .dialog-head {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--border); background: #f8fafc; }}
    .dialog-title {{ margin: 0; font-size: 1rem; }}
    .dialog-close {{ border: 1px solid #94a3b8; background: #fff; border-radius: 8px; padding: 4px 10px; cursor: pointer; }}
    .dialog-body {{ padding: 16px; }}
    .compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .compare-card {{ border: 1px solid var(--border); border-radius: 10px; background: #fff; padding: 10px; }}
    .compare-card img {{ width: 100%; max-height: 70vh; object-fit: contain; background: #0f172a; border-radius: 8px; }}
    .compare-caption {{ margin-top: 8px; font-weight: 600; }}
    .compare-path {{ margin-top: 2px; color: #334155; font-size: 0.82rem; word-break: break-all; font-family: 'Consolas', 'Courier New', monospace; }}
    .dialog-meta {{ margin-top: 10px; color: #334155; font-size: 0.9rem; }}
    @media (max-width: 780px) {{
      .image-cell {{ grid-template-columns: 72px 1fr; min-width: 260px; }}
      .thumb {{ width: 72px; height: 48px; }}
      th, td {{ padding: 8px; }}
      .compare-grid {{ grid-template-columns: 1fr; }}
    }}
    .foot {{ margin-top: 10px; color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <h1>Similar Images Scan Report</h1>
      <div class=\"meta\">Scanned folders:</div>
      <ul class=\"scan-list\">{scanned_folders_html}</ul>
      <div class=\"meta\">Generated at: {generated_at}</div>
      <div class=\"meta\">Thresholds - similar: {similar_threshold:.2f}, duplicate: {duplicate_threshold:.2f}</div>
      <div class=\"meta\">Weights - histogram: {weights.histogram:.2f}, pHash: {weights.phash:.2f}, HOG: {weights.hog:.2f}, ORB: {weights.orb:.2f}, SSIM: {weights.ssim:.2f}, Edge: {weights.edge:.2f} (auto-normalized)</div>
      <div class=\"stats\">
        <div class=\"stat\"><div class=\"k\">Images loaded</div><div class=\"v\">{loaded_count}</div></div>
        <div class=\"stat\"><div class=\"k\">Images skipped</div><div class=\"v\">{skipped_count}</div></div>
        <div class=\"stat\"><div class=\"k\">Duplicate pairs</div><div class=\"v\">{duplicates}</div></div>
        <div class=\"stat\"><div class=\"k\">Similar pairs</div><div class=\"v\">{similars}</div></div>
        <div class=\"stat\"><div class=\"k\">Different pairs</div><div class=\"v\">{differents}</div></div>
      </div>
    </section>

    <section class=\"table-wrap\">
      <table>
        <thead>
          <tr>
            <th>Left Image</th>
            <th>Right Image</th>
            <th class=\"col-score\">Score</th>
            <th class=\"col-classifier\">Classifier</th>
          </tr>
        </thead>
        <tbody>
          {table_body}
        </tbody>
      </table>
    </section>
    <dialog id=\"compareDialog\" class=\"compare-dialog\">
      <div class=\"dialog-head\">
        <h2 id=\"dialogTitle\" class=\"dialog-title\">Image Comparison</h2>
        <button id=\"dialogClose\" type=\"button\" class=\"dialog-close\">Close</button>
      </div>
      <div class=\"dialog-body\">
        <div class=\"compare-grid\">
          <figure class=\"compare-card\">
            <img id=\"dialogLeftImg\" src=\"\" alt=\"Left image\" />
            <figcaption class=\"compare-caption\" id=\"dialogLeftName\"></figcaption>
            <div class=\"compare-path\" id=\"dialogLeftPath\"></div>
          </figure>
          <figure class=\"compare-card\">
            <img id=\"dialogRightImg\" src=\"\" alt=\"Right image\" />
            <figcaption class=\"compare-caption\" id=\"dialogRightName\"></figcaption>
            <div class=\"compare-path\" id=\"dialogRightPath\"></div>
          </figure>
        </div>
        <div class=\"dialog-meta\" id=\"dialogMeta\"></div>
      </div>
    </dialog>
    <div class=\"foot\">Thumbnails are shown for similar and duplicate rows. Click classifier chips for a larger side-by-side comparison.</div>
  </main>
  <script>
    (function () {{
      const imageRegistry = {image_registry_json};
      const dialog = document.getElementById('compareDialog');
      const dialogTitle = document.getElementById('dialogTitle');
      const dialogClose = document.getElementById('dialogClose');
      const leftImg = document.getElementById('dialogLeftImg');
      const rightImg = document.getElementById('dialogRightImg');
      const leftName = document.getElementById('dialogLeftName');
      const rightName = document.getElementById('dialogRightName');
      const leftPath = document.getElementById('dialogLeftPath');
      const rightPath = document.getElementById('dialogRightPath');
      const dialogMeta = document.getElementById('dialogMeta');

      document.querySelectorAll('.js-thumb').forEach((img) => {{
        const imageId = Number.parseInt(img.dataset.imgId || '-1', 10);
        if (!Number.isNaN(imageId) && imageRegistry[imageId]) {{
          img.src = imageRegistry[imageId].uri;
          img.alt = imageRegistry[imageId].name;
        }}
      }});

      document.querySelectorAll('.compare-btn').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const leftId = Number.parseInt(btn.dataset.leftId || '-1', 10);
          const rightId = Number.parseInt(btn.dataset.rightId || '-1', 10);
          if (Number.isNaN(leftId) || Number.isNaN(rightId)) {{
            return;
          }}

          const left = imageRegistry[leftId];
          const right = imageRegistry[rightId];
          if (!left || !right) {{
            return;
          }}

          const score = btn.dataset.score || '';
          const classifier = btn.dataset.classifier || '';

          leftImg.src = left.uri;
          rightImg.src = right.uri;
          leftName.textContent = left.name;
          rightName.textContent = right.name;
          leftPath.textContent = left.path;
          rightPath.textContent = right.path;
          dialogTitle.textContent = `${{classifier.charAt(0).toUpperCase() + classifier.slice(1)}} comparison`;
          dialogMeta.textContent = `Score: ${{score}} | Classifier: ${{classifier}}`;

          if (typeof dialog.showModal === 'function') {{
            dialog.showModal();
          }}
        }});
      }});

      dialogClose.addEventListener('click', () => {{
        dialog.close();
      }});

      dialog.addEventListener('click', (event) => {{
        const rect = dialog.getBoundingClientRect();
        const inside = (
          event.clientX >= rect.left &&
          event.clientX <= rect.right &&
          event.clientY >= rect.top &&
          event.clientY <= rect.bottom
        );
        if (!inside) {{
          dialog.close();
        }}
      }});
    }})();
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
