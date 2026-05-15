from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .models import PairResult


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


def build_html_report(
    scanned_folder: Path,
    output_path: Path,
    results: list[PairResult],
    loaded_count: int,
    skipped_count: int,
    similar_threshold: float,
    duplicate_threshold: float,
) -> None:
    duplicates, similars, differents = _summary_counts(results)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = []
    for item in results:
        rows.append(
            "".join(
                [
                    f"<tr class='{_row_class(item.classifier)}'>",
                    f"<td>{escape(item.left.name)}</td>",
                    f"<td>{escape(str(item.left.path))}</td>",
                    f"<td>{escape(item.right.name)}</td>",
                    f"<td>{escape(str(item.right.path))}</td>",
                    f"<td>{item.score:.4f}</td>",
                    f"<td>{escape(item.classifier)}</td>",
                    "</tr>",
                ]
            )
        )

    table_body = "\n".join(rows) if rows else "<tr><td colspan='7'>No comparable image pairs found.</td></tr>"

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
    .stats {{ margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }}
    .stat .k {{ color: var(--muted); font-size: 0.85rem; }}
    .stat .v {{ font-size: 1.2rem; font-weight: 700; }}
    .table-wrap {{ margin-top: 18px; overflow: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 960px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.92rem; }}
    th {{ background: #f1f5f9; position: sticky; top: 0; z-index: 1; }}
    tr.duplicate td {{ background: var(--dup); }}
    tr.similar td {{ background: var(--sim); }}
    tr.different td {{ background: var(--diff); }}
    .foot {{ margin-top: 10px; color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <h1>Similar Images Scan Report</h1>
      <div class=\"meta\">Scanned folder: {escape(str(scanned_folder))}</div>
      <div class=\"meta\">Generated at: {generated_at}</div>
      <div class=\"meta\">Thresholds - similar: {similar_threshold:.2f}, duplicate: {duplicate_threshold:.2f}</div>
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
            <th>Left Name</th>
            <th>Left Path</th>
            <th>Right Name</th>
            <th>Right Path</th>
            <th>Score</th>
            <th>Classifier</th>
          </tr>
        </thead>
        <tbody>
          {table_body}
        </tbody>
      </table>
    </section>
    <div class=\"foot\">Classifier values: duplicate, similar, different.</div>
  </main>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
