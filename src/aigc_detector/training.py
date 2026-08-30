"""Shared training and evaluation utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    model.eval()
    labels, probabilities, records = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["pixel_values"].to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            batch_labels = batch["label"].numpy()
            labels.extend(batch_labels.tolist())
            probabilities.extend(scores.tolist())
            records.extend(
                {"image_path": path, "label": int(label), "pred": float(score)}
                for path, label, score in zip(batch["image_path"], batch_labels, scores, strict=True)
            )
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }
    return metrics, records


def save_checkpoint(path: str | Path, model: nn.Module, model_name: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "model_state": model.state_dict()}, destination)
