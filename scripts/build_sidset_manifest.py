#!/usr/bin/env python3
"""Build deterministic row-level manifests for the SID_Set parquet shards.

Reads only the metadata columns, never the image bytes, so it runs in seconds.
Each manifest row locates one image by shard path and row index.
"""
import argparse
import csv
import glob
import os

import pyarrow.parquet as pq

LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}
META_COLUMNS = ["img_id", "width", "height", "label"]


def shard_split(path: str) -> str:
    """Infer the split from a filename like train-00000-of-00249.parquet."""
    return os.path.basename(path).split("-")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/SID_Set/data")
    ap.add_argument("--out-dir", default="data/manifests")
    ap.add_argument(
        "--keep-tampered",
        action="store_true",
        help="Include label 2 rows, mapped to is_aigc=1. Excluded by default.",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet shards under {args.data_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    writers, handles, counts = {}, {}, {}

    for path in files:
        split = shard_split(path)
        if split not in writers:
            fh = open(os.path.join(args.out_dir, f"sidset_{split}.csv"), "w", newline="")
            w = csv.writer(fh)
            w.writerow(
                ["shard", "row", "img_id", "label", "label_name", "is_aigc", "width", "height"]
            )
            handles[split], writers[split] = fh, w
            counts[split] = {}

        table = pq.read_table(path, columns=META_COLUMNS)
        rel = os.path.relpath(path, start=os.path.dirname(args.data_root))
        for i, rec in enumerate(table.to_pylist()):
            label = rec["label"]
            if label == 2 and not args.keep_tampered:
                continue
            name = LABEL_NAMES.get(label, str(label))
            writers[split].writerow(
                [
                    rel,
                    i,
                    rec["img_id"],
                    label,
                    name,
                    1 if label in (1, 2) else 0,
                    rec["width"],
                    rec["height"],
                ]
            )
            counts[split][name] = counts[split].get(name, 0) + 1

    for split, fh in handles.items():
        fh.close()
        total = sum(counts[split].values())
        print(f"{split}: {total} rows  {counts[split]}")


if __name__ == "__main__":
    main()
