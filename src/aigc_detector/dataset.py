"""Streaming dataset over the SID_Set parquet shards.

Random access into parquet is expensive, so this iterates shard by shard and
shuffles within each one. Images are cropped at native resolution rather than
resized: SID_Set synthetic images are all 1024x1024 while real ones are not,
so any whole-image resize leaks resolution as a label shortcut.
"""
import csv
import io
import math
import random

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from . import transforms as T

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CROP = 224


def load_manifest(path):
    """Group manifest rows by shard, keeping label and row index."""
    shards = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            shards.setdefault(row["shard"], []).append(
                (int(row["row"]), int(row["is_aigc"]))
            )
    return shards


def to_tensor(img, rng):
    """Crop a CROP-sized patch at native scale and normalise for CLIP."""
    w, h = img.size
    if w < CROP or h < CROP:
        scale = CROP / min(w, h)
        img = img.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)
        w, h = img.size
    left = rng.randint(0, w - CROP)
    top = rng.randint(0, h - CROP)
    patch = img.crop((left, top, left + CROP, top + CROP))
    arr = torch.frombuffer(bytearray(patch.tobytes()), dtype=torch.uint8)
    arr = arr.view(CROP, CROP, 3).permute(2, 0, 1).float().div_(255.0)
    return (arr - CLIP_MEAN) / CLIP_STD


class SidSet(IterableDataset):
    """Yields (pixel_values, label, transform_name) tuples."""

    def __init__(self, manifest, data_root, augment=True, seed=0, limit_shards=None):
        self.shards = load_manifest(manifest)
        self.keys = sorted(self.shards)
        if limit_shards:
            self.keys = self.keys[:limit_shards]
        self.data_root = data_root
        self.augment = augment
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        rng = random.Random(self.seed * 1000 + wid)

        keys = list(self.keys)
        rng.shuffle(keys)
        for key in keys[wid::nworkers]:
            rows = list(self.shards[key])
            rng.shuffle(rows)
            wanted = sorted(r for r, _ in rows)
            table = pq.read_table(f"{self.data_root}/{key}", columns=["image"])
            col = table.column("image").to_pylist()
            labels = dict(rows)
            for idx in wanted:
                rec = col[idx]
                if rec is None:
                    continue
                try:
                    img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
                except Exception:
                    continue
                name = T.sample_train_transform(rng) if self.augment else "clean"
                img = T.named(img, name, rng)
                yield to_tensor(img, rng), labels[idx], name
            del table, col
