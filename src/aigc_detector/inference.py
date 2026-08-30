from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor

from .model import ClipBinaryDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_detector(checkpoint_path: Path, device: torch.device) -> tuple[ClipBinaryDetector, CLIPImageProcessor]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = payload["model_name"]
    model = ClipBinaryDetector(model_name, freeze_encoder=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, CLIPImageProcessor.from_pretrained(model_name)


@torch.inference_mode()
def predict_image(model: ClipBinaryDetector, processor: CLIPImageProcessor, path: Path, device: torch.device) -> float:
    with Image.open(path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    logits = model(inputs["pixel_values"].to(device))
    return float(torch.sigmoid(logits).item())
