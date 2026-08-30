#!/usr/bin/env python3
"""Train the CLIP-based binary AIGC baseline from a YAML configuration."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_detector.data import ManifestImageDataset  # noqa: E402
from aigc_detector.model import ClipBinaryDetector  # noqa: E402
from aigc_detector.training import evaluate, save_checkpoint  # noqa: E402
from aigc_detector.transforms import random_challenge_transform  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained(config["model_name"])
    train_set = ManifestImageDataset(config["data"]["train_manifest"], processor, random_challenge_transform)
    val_set = ManifestImageDataset(config["data"]["val_manifest"], processor)
    loader_args = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = ClipBinaryDetector(config["model_name"], config["freeze_encoder"]).to(device)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best_auroc = float("-inf")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["pixel_values"].to(device))
            loss = loss_fn(logits, batch["label"].float().to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_auroc={metrics['auroc']:.4f} val_f1={metrics['f1']:.4f}")
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            save_checkpoint(ROOT / config["output"]["checkpoint"], model, config["model_name"])


if __name__ == "__main__":
    main()
