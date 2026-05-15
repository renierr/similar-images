from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
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
    output: Path = typer.Option(Path("report.html"), "--output", "-o", help="HTML report path."),
) -> None:
    if duplicate_threshold < similar_threshold:
        raise typer.BadParameter("duplicate-threshold must be >= similar-threshold")

    weights = SimilarityWeights(
        histogram=histogram_weight,
        phash=phash_weight,
        hog=hog_weight,
        orb=orb_weight,
    )
    if weights.total() <= 0.0:
        raise typer.BadParameter(
            "At least one extraction weight must be > 0. "
            "Use --histogram-weight, --phash-weight, or --hog-weight."
        )

    resolved_folders = [folder.resolve() for folder in folders]
    records = scan_images_from_folders(resolved_folders, recursive=recursive)
    if not records:
        console.print("No supported images found in provided folders.")
        raise typer.Exit(code=1)

    console.print(f"Found {len(records)} image files. Building features and comparing...")
    results, loaded_records = compare_all(
        records=records,
        similar_threshold=similar_threshold,
        duplicate_threshold=duplicate_threshold,
        weights=weights,
    )

    skipped_count = len(records) - len(loaded_records)
    build_html_report(
        scanned_folders=resolved_folders,
        output_path=output.resolve(),
        results=results,
        loaded_count=len(loaded_records),
        skipped_count=skipped_count,
        similar_threshold=similar_threshold,
        duplicate_threshold=duplicate_threshold,
        weights=weights,
    )

    table = Table(title="Scan Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total files", str(len(records)))
    table.add_row("Images loaded", str(len(loaded_records)))
    table.add_row("Images skipped", str(skipped_count))
    table.add_row("Pair comparisons", str(len(results)))
    table.add_row("Histogram weight", f"{weights.histogram:.2f}")
    table.add_row("pHash weight", f"{weights.phash:.2f}")
    table.add_row("HOG weight", f"{weights.hog:.2f}")
    table.add_row("ORB weight", f"{weights.orb:.2f}")
    table.add_row("Report", str(output.resolve()))
    console.print(table)


def main() -> None:
    app()


@app.command("version")
def version() -> None:
    console.print("similar-images 0.1.0")


if __name__ == "__main__":
    main()
