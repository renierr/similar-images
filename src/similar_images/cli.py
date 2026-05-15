from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .classifier import compare_all
from .io import scan_images_from_folders
from .report import build_html_report
from .similarity import SimilarityWeights

app = typer.Typer(help="Scan a folder and classify similar images.")
console = Console()


@app.command("scan")
def scan(
    folders: list[Path] = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan nested folders too."),
    similar_threshold: float = typer.Option(0.82, min=0.0, max=1.0, help="Threshold for 'similar'."),
    duplicate_threshold: float = typer.Option(0.96, min=0.0, max=1.0, help="Threshold for 'duplicate'."),
    histogram_weight: float = typer.Option(0.4, min=0.0, max=1.0, help="Weight for histogram similarity."),
    phash_weight: float = typer.Option(0.2, min=0.0, max=1.0, help="Weight for pHash similarity."),
    hog_weight: float = typer.Option(0.4, min=0.0, max=1.0, help="Weight for HOG similarity."),
    orb_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Weight for ORB keypoint match similarity."),
    ssim_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Weight for grayscale SSIM similarity."),
    edge_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Weight for edge-structure similarity."),
    report_min_score: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="Minimum score to include rows in HTML report.",
    ),
    report_max_rows: int = typer.Option(
        0,
        min=0,
        help="Maximum rows in HTML report (0 = unlimited).",
    ),
    output: Path = typer.Option(Path("report.html"), "--output", "-o", help="HTML report path."),
) -> None:
    if duplicate_threshold < similar_threshold:
        raise typer.BadParameter("duplicate-threshold must be >= similar-threshold")

    weights = SimilarityWeights(
        histogram=histogram_weight,
        phash=phash_weight,
        hog=hog_weight,
        orb=orb_weight,
        ssim=ssim_weight,
        edge=edge_weight,
    )
    if weights.total() <= 0.0:
        raise typer.BadParameter(
            "At least one extraction weight must be > 0. "
            "Use --histogram-weight, --phash-weight, --hog-weight, --orb-weight, --ssim-weight, or --edge-weight."
        )

    resolved_folders = [folder.resolve() for folder in folders]
    records = scan_images_from_folders(resolved_folders, recursive=recursive)
    if not records:
        console.print("No supported images found in provided folders.")
        raise typer.Exit(code=1)

    console.print(f"Found {len(records)} image files. Building features and comparing...")

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        feature_task = progress.add_task("Extracting features", total=len(records))
        compare_task = progress.add_task("Comparing pairs", total=None)

        def on_feature_progress(done: int, total: int) -> None:
            progress.update(feature_task, completed=done, total=total)

        def on_compare_start(total_pairs: int) -> None:
            progress.update(compare_task, total=total_pairs, completed=0)

        def on_compare_progress(done: int, total: int) -> None:
            progress.update(compare_task, completed=done, total=total)

        results, loaded_records = compare_all(
            records=records,
            similar_threshold=similar_threshold,
            duplicate_threshold=duplicate_threshold,
            weights=weights,
            on_feature_progress=on_feature_progress,
            on_compare_start=on_compare_start,
            on_compare_progress=on_compare_progress,
        )

    skipped_count = len(records) - len(loaded_records)

    filtered_results = [r for r in results if r.score >= report_min_score]
    if report_max_rows > 0:
        filtered_results = filtered_results[:report_max_rows]

    build_html_report(
        scanned_folders=resolved_folders,
        output_path=output.resolve(),
        results=filtered_results,
        loaded_count=len(loaded_records),
        skipped_count=skipped_count,
        similar_threshold=similar_threshold,
        duplicate_threshold=duplicate_threshold,
        weights=weights,
        report_min_score=report_min_score,
        report_max_rows=report_max_rows,
    )

    table = Table(title="Scan Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total files", str(len(records)))
    table.add_row("Images loaded", str(len(loaded_records)))
    table.add_row("Images skipped", str(skipped_count))
    table.add_row("Pair comparisons", str(len(results)))
    table.add_row("Report rows", str(len(filtered_results)))
    table.add_row("Report min score", f"{report_min_score:.2f}")
    table.add_row("Report max rows", "unlimited" if report_max_rows == 0 else str(report_max_rows))
    table.add_row("Histogram weight", f"{weights.histogram:.2f}")
    table.add_row("pHash weight", f"{weights.phash:.2f}")
    table.add_row("HOG weight", f"{weights.hog:.2f}")
    table.add_row("ORB weight", f"{weights.orb:.2f}")
    table.add_row("SSIM weight", f"{weights.ssim:.2f}")
    table.add_row("Edge weight", f"{weights.edge:.2f}")
    table.add_row("Report", str(output.resolve()))
    console.print(table)


def main() -> None:
    app()


@app.command("version")
def version() -> None:
    console.print("similar-images 0.1.0")


if __name__ == "__main__":
    main()
