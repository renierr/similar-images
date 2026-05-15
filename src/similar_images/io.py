from __future__ import annotations

from pathlib import Path

from .models import ImageRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}


def scan_images(folder: Path, recursive: bool) -> list[ImageRecord]:
    pattern = "**/*" if recursive else "*"
    records: list[ImageRecord] = []

    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        records.append(ImageRecord(path=path.resolve(), name=path.name))

    records.sort(key=lambda x: str(x.path).lower())
    return records


def scan_images_from_folders(folders: list[Path], recursive: bool) -> list[ImageRecord]:
    deduped: dict[Path, ImageRecord] = {}
    for folder in folders:
        for record in scan_images(folder=folder, recursive=recursive):
            deduped[record.path] = record

    records = list(deduped.values())
    records.sort(key=lambda x: str(x.path).lower())
    return records
