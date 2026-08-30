#!/usr/bin/env python3
"""Create deterministic CIFAKE CSV manifests without committing raw images."""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/CIFAKE"
OUT = ROOT / "data/manifests"
SEED = 42


def rows(split: str) -> list[dict[str, str | int]]:
    records = []
    for folder, label in (("REAL", 0), ("FAKE", 1)):
        directory = RAW / split / folder
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                records.append({"image_path": str(path), "label": label})
    return records


def write(name: str, records: list[dict[str, str | int]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"cifake_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    train = rows("train")
    test = rows("test")
    if not train or not test:
        raise SystemExit(f"Expected images under {RAW}/train and {RAW}/test")
    random.Random(SEED).shuffle(train)
    cut = int(len(train) * 0.9)
    write("train", train[:cut])
    write("val", train[cut:])
    write("test", test)


if __name__ == "__main__":
    main()
