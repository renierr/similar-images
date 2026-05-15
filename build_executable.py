from __future__ import annotations

import platform
import shutil
from pathlib import Path

import PyInstaller.__main__


def main() -> None:
    project_root = Path(__file__).resolve().parent
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    entry_script = project_root / "run_similar_images.py"
    args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--noconsole",
        "--name",
        "similar-images",
        "--paths",
        str(project_root / "src"),
        "--collect-all",
        "customtkinter",
        str(entry_script),
    ]

    PyInstaller.__main__.run(args)

    exe_name = "similar-images.exe" if platform.system() == "Windows" else "similar-images"
    exe_path = dist_dir / exe_name
    print(f"Executable generated: {exe_path}")


if __name__ == "__main__":
    main()
