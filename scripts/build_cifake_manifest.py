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
    train_split, val_split = [], []
    for label in (0, 1):
        class_records = [record for record in train if record["label"] == label]
        random.Random(SEED + label).shuffle(class_records)
        cut = int(len(class_records) * 0.9)
        train_split.extend(class_records[:cut])
        val_split.extend(class_records[cut:])
    random.Random(SEED).shuffle(train_split)
    random.Random(SEED).shuffle(val_split)
    write("train", train_split)
    write("val", val_split)
    write("test", test)


if __name__ == "__main__":
    main()
