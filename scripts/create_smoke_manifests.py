#!/usr/bin/env python3
"""Create small class-balanced manifests for a fast end-to-end smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def balanced_subset(source: Path, per_class: int) -> pd.DataFrame:
    data = pd.read_csv(source)
    subset = data.groupby("label", group_keys=False).head(per_class)
    if subset.groupby("label").size().min() < per_class:
        raise ValueError(f"{source} does not contain {per_class} images per class")
    return subset.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    for split in ("train", "val"):
        source = args.manifest_dir / f"cifake_{split}.csv"
        destination = args.manifest_dir / f"smoke_{split}.csv"
        subset = balanced_subset(source, args.per_class)
        subset.to_csv(destination, index=False)
        print(f"Wrote {destination}: {len(subset)} images")


if __name__ == "__main__":
    main()
