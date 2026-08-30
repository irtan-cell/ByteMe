#!/usr/bin/env python3
"""Validate CIFAKE folder layout, class balance, and image readability."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_FOLDERS = (("train", "REAL"), ("train", "FAKE"), ("test", "REAL"), ("test", "FAKE"))


def image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate(root: Path) -> dict:
    counts: Counter[str] = Counter()
    unreadable: list[str] = []
    for split, label in EXPECTED_FOLDERS:
        directory = root / split / label
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing required folder: {directory}")
        paths = image_paths(directory)
        counts[f"{split}/{label}"] = len(paths)
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                unreadable.append(f"{path}: {error}")
    return {"root": str(root), "counts": dict(counts), "total_images": sum(counts.values()), "unreadable_images": unreadable}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/CIFAKE"))
    parser.add_argument("--report", type=Path, default=Path("outputs/data_validation/cifake_validation.json"))
    args = parser.parse_args()
    report = validate(args.data_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["unreadable_images"]:
        raise SystemExit("Dataset contains unreadable images; see validation report.")


if __name__ == "__main__":
    main()
