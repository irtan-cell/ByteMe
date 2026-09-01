#!/usr/bin/env python3
"""Evaluate the detector across every transformation in the challenge grid.

Applies one condition to the whole evaluation set at a time, so clean and
transformed metrics are measured on identical images. Writes per-condition
metrics and per-image predictions for error analysis.
"""
import argparse
import csv
import io
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aigc_detector import transforms as T
from aigc_detector.dataset import CLIP_MEAN, CLIP_STD, CROP
from aigc_detector.model import Detector


def load_rows(manifest, limit, seed):
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def group_by_shard(rows):
    shards = {}
    for r in rows:
        shards.setdefault(r["shard"], []).append((int(r["row"]), int(r["is_aigc"]), r["img_id"]))
    return shards


def center_tensor(img):
    w, h = img.size
    if w < CROP or h < CROP:
        s = CROP / min(w, h)
        img = img.resize((max(CROP, int(w * s) + 1), max(CROP, int(h * s) + 1)), Image.BICUBIC)
        w, h = img.size
    patch = img.crop(((w - CROP) // 2, (h - CROP) // 2,
                      (w - CROP) // 2 + CROP, (h - CROP) // 2 + CROP))
    arr = torch.frombuffer(bytearray(patch.tobytes()), dtype=torch.uint8)
    arr = arr.view(CROP, CROP, 3).permute(2, 0, 1).float().div_(255.0)
    return (arr - CLIP_MEAN) / CLIP_STD


def metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    auroc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else float("nan")
    acc = float(((scores >= 0.5).astype(int) == labels).mean())
    # Threshold at 1% false positive rate on real images.
    real = np.sort(scores[labels == 0])
    thr = float(real[int(0.99 * len(real))]) if len(real) else 0.5
    tpr = float((scores[labels == 1] >= thr).mean()) if (labels == 1).any() else float("nan")
    return {"auroc": round(auroc, 4), "acc@0.5": round(acc, 4),
            "tpr@1%fpr": round(tpr, 4), "threshold@1%fpr": round(thr, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/sidset_validation.csv")
    ap.add_argument("--data-root", default="data/raw/SID_Set")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Detector(unfreeze_last=ckpt.get("args", {}).get("unfreeze_last", 0))
    # Head-only checkpoints omit the frozen encoder, which loads from the hub.
    model.load_state_dict(ckpt["model"], strict=not ckpt.get("head_only"))
    model.cuda().eval()

    rows = load_rows(args.manifest, args.limit, args.seed)
    shards = group_by_shard(rows)
    print(f"{len(rows)} images across {len(shards)} shards", flush=True)

    # Decode every image once, then reuse the decoded copies for each condition.
    images, labels, ids = [], [], []
    for shard, entries in shards.items():
        table = pq.read_table(f"{args.data_root}/{shard}", columns=["image"])
        col = table.column("image").to_pylist()
        for idx, lab, img_id in entries:
            rec = col[idx]
            if rec is None:
                continue
            try:
                images.append(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
                labels.append(lab)
                ids.append(img_id)
            except Exception:
                continue
        del table, col
    print(f"decoded {len(images)} images", flush=True)

    rng = random.Random(args.seed)
    results, per_image = {}, {}

    for name in T.eval_grid():
        scores = []
        buf = []
        for img in images:
            buf.append(center_tensor(T.named(img, name, rng)))
            if len(buf) == args.batch_size:
                x = torch.stack(buf).cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    scores += model(x).float().sigmoid().cpu().tolist()
                buf = []
        if buf:
            x = torch.stack(buf).cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                scores += model(x).float().sigmoid().cpu().tolist()
        results[name] = metrics(labels, scores)
        per_image[name] = scores
        print(f"{name:12s} {results[name]}", flush=True)

    clean = results["clean"]["auroc"]
    for name in results:
        results[name]["auroc_gap_vs_clean"] = round(clean - results[name]["auroc"], 4)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "robustness.json"), "w") as fh:
        json.dump({"n_images": len(images), "checkpoint": args.checkpoint,
                   "conditions": results}, fh, indent=2)

    with open(os.path.join(args.out_dir, "robustness.md"), "w") as fh:
        fh.write(f"# Robustness ({len(images)} validation images)\n\n")
        fh.write("| Condition | AUROC | Acc@0.5 | TPR@1%FPR | AUROC gap |\n")
        fh.write("|---|---|---|---|---|\n")
        for name, m in results.items():
            fh.write(f"| {name} | {m['auroc']} | {m['acc@0.5']} | "
                     f"{m['tpr@1%fpr']} | {m['auroc_gap_vs_clean']} |\n")

    with open(os.path.join(args.out_dir, "per_image_scores.json"), "w") as fh:
        json.dump({"img_id": ids, "label": labels, "scores": per_image}, fh)

    print(f"\nwrote {args.out_dir}/robustness.md", flush=True)


if __name__ == "__main__":
    main()
