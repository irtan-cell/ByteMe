#!/usr/bin/env python3
"""Score every image in a directory for the probability it is AI-generated.

Writes the submission format: a JSON array of {"image_path", "pred"} objects.
Scores are averaged over several 224 crops, since the model reads patches at
native resolution rather than a resized whole image.
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aigc_detector.dataset import CLIP_MEAN, CLIP_STD, CROP
from aigc_detector.model import Detector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in EXTENSIONS:
                found.append(os.path.join(dirpath, n))
    return sorted(found)


def crops(img, n_crops):
    """Take a centre crop plus a grid of others, all at native resolution."""
    w, h = img.size
    if w < CROP or h < CROP:
        scale = CROP / min(w, h)
        img = img.resize((max(CROP, int(w * scale) + 1), max(CROP, int(h * scale) + 1)),
                         Image.BICUBIC)
        w, h = img.size
    positions = [((w - CROP) // 2, (h - CROP) // 2)]
    if n_crops > 1:
        steps = max(1, int(n_crops ** 0.5))
        xs = [round(i * (w - CROP) / max(1, steps - 1)) for i in range(steps)]
        ys = [round(i * (h - CROP) / max(1, steps - 1)) for i in range(steps)]
        positions += [(x, y) for x in xs for y in ys]
    out = []
    for x, y in positions[:n_crops]:
        patch = img.crop((x, y, x + CROP, y + CROP))
        arr = torch.frombuffer(bytearray(patch.tobytes()), dtype=torch.uint8)
        arr = arr.view(CROP, CROP, 3).permute(2, 0, 1).float().div_(255.0)
        out.append((arr - CLIP_MEAN) / CLIP_STD)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image directory")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--output", default="outputs/predictions.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crops", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Detector(unfreeze_last=ckpt.get("args", {}).get("unfreeze_last", 0))
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    paths = find_images(args.input)
    print(f"{len(paths)} images found", flush=True)

    results, batch, batch_paths = [], [], []

    def flush():
        if not batch:
            return
        counts = [t.shape[0] for t in batch]
        x = torch.cat(batch).to(args.device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = model(x).float().sigmoid().cpu()
        i = 0
        for p, c in zip(batch_paths, counts):
            results.append({"image_path": p, "pred": round(probs[i:i + c].mean().item(), 6)})
            i += c
        batch.clear()
        batch_paths.clear()

    for n, path in enumerate(paths, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"skipped {path}: {exc}", flush=True)
            continue
        batch.append(crops(img, args.crops))
        batch_paths.append(path)
        if len(batch) >= args.batch_size:
            flush()
        if n % 500 == 0:
            print(f"  {n}/{len(paths)}", flush=True)
    flush()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {len(results)} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
