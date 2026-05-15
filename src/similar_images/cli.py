from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .classifier import compare_all
from .io import scan_images
from .report import build_html_report

app = typer.Typer(help="Scan a folder and classify similar images.")
console = Console()


@app.command("scan")
def scan(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan nested folders too."),
    similar_threshold: float = typer.Option(0.82, min=0.0, max=1.0, help="Threshold for 'similar'."),
    duplicate_threshold: float = typer.Option(0.96, min=0.0, max=1.0, help="Threshold for 'duplicate'."),
    output: Path = typer.Option(Path("report.html"), "--output", "-o", help="HTML report path."),
) -> None:
    if duplicate_threshold < similar_threshold:
        raise typer.BadParameter("duplicate-threshold must be >= similar-threshold")

    records = scan_images(folder.resolve(), recursive=recursive)
    if not records:
        console.print("No supported images found in folder.")
        raise typer.Exit(code=1)

    console.print(f"Found {len(records)} image files. Building features and comparing...")
    results, loaded_records = compare_all(
        records=records,
        similar_threshold=similar_threshold,
        duplicate_threshold=duplicate_threshold,
    )

    skipped_count = len(records) - len(loaded_records)
    build_html_report(
        scanned_folder=folder.resolve(),
        output_path=output.resolve(),
        results=results,
        loaded_count=len(loaded_records),
        skipped_count=skipped_count,
        similar_threshold=similar_threshold,
        duplicate_threshold=duplicate_threshold,
    )

    table = Table(title="Scan Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total files", str(len(records)))
    table.add_row("Images loaded", str(len(loaded_records)))
    table.add_row("Images skipped", str(skipped_count))
    table.add_row("Pair comparisons", str(len(results)))
    table.add_row("Report", str(output.resolve()))
    console.print(table)


def main() -> None:
    app()


@app.command("version")
def version() -> None:
    console.print("similar-images 0.1.0")


if __name__ == "__main__":
    main()
