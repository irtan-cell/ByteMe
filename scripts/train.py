#!/usr/bin/env python3
"""Train the AIGC detector head on SID_Set."""
import argparse
import os
import sys
import time

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aigc_detector.dataset import SidSet
from aigc_detector.model import Detector


def evaluate(model, loader, device, max_batches):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            scores += logits.float().sigmoid().cpu().tolist()
            labels += y.tolist()
    model.train()
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return roc_auc_score(labels, scores), len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/SID_Set")
    ap.add_argument("--train-manifest", default="data/manifests/sidset_train.csv")
    ap.add_argument("--val-manifest", default="data/manifests/sidset_validation.csv")
    ap.add_argument("--out", default="checkpoints/best.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--unfreeze-last", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()

    device = "cuda"
    model = Detector(unfreeze_last=args.unfreeze_last).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    train_ds = SidSet(args.train_manifest, args.data_root, augment=True, seed=1,
                      limit_shards=args.limit_shards)
    val_ds = SidSet(args.val_manifest, args.data_root, augment=False, seed=2, limit_shards=8)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    lossfn = torch.nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, step, running, t0 = 0.0, 0, 0.0, time.time()

    while step < args.steps:
        for x, y, _ in train_dl:
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            running += loss.item()
            step += 1

            if step % 50 == 0:
                rate = step / (time.time() - t0)
                print(f"step {step} loss {running / 50:.4f} {rate:.1f} it/s", flush=True)
                running = 0.0

            if step % args.eval_every == 0:
                auroc, n = evaluate(model, val_dl, device, args.eval_batches)
                print(f"  eval step {step}: AUROC {auroc:.4f} on {n} images", flush=True)
                if auroc > best:
                    best = auroc
                    torch.save({"model": model.state_dict(), "step": step,
                                "auroc": auroc, "args": vars(args)}, args.out)
                    print(f"  saved {args.out}", flush=True)

    print(f"done. best AUROC {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
