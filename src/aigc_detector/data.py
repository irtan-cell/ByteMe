"""Manifest-backed datasets for reproducible AIGC experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class ManifestImageDataset(Dataset):
    """Loads labelled RGB images from a CSV with image_path and label columns."""

    def __init__(
        self,
        manifest_path: str | Path,
        processor: CLIPImageProcessor,
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = pd.read_csv(self.manifest_path)
        required = {"image_path", "label"}
        if not required.issubset(self.records.columns):
            raise ValueError(f"{self.manifest_path} must contain {sorted(required)}")
        self.processor = processor
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records.iloc[index]
        image_path = Path(row.image_path)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "label": int(row.label), "image_path": str(image_path)}
