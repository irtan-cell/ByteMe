"""Local Flask interface for the ByteMe image detector."""

from __future__ import annotations

import io
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aigc_detector.model import Detector  # noqa: E402
CHECKPOINT = ROOT / "checkpoints" / "head_only.pt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CROP_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

_model: Detector | None = None
_device: torch.device | None = None
_model_lock = Lock()


def make_crops(image: Image.Image, count: int = 5) -> torch.Tensor:
    """Create the same native-resolution crop set used by directory inference."""
    width, height = image.size
    if width < CROP_SIZE or height < CROP_SIZE:
        scale = CROP_SIZE / min(width, height)
        image = image.resize(
            (
                max(CROP_SIZE, int(width * scale) + 1),
                max(CROP_SIZE, int(height * scale) + 1),
            ),
            Image.Resampling.BICUBIC,
        )
        width, height = image.size

    positions = [((width - CROP_SIZE) // 2, (height - CROP_SIZE) // 2)]
    if count > 1:
        steps = max(1, int(count**0.5))
        xs = [
            round(index * (width - CROP_SIZE) / max(1, steps - 1))
            for index in range(steps)
        ]
        ys = [
            round(index * (height - CROP_SIZE) / max(1, steps - 1))
            for index in range(steps)
        ]
        positions.extend((x, y) for x in xs for y in ys)

    tensors = []
    for x, y in positions[:count]:
        patch = image.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
        tensor = torch.frombuffer(bytearray(patch.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(CROP_SIZE, CROP_SIZE, 3).permute(2, 0, 1)
        tensor = tensor.float().div_(255.0)
        tensors.append((tensor - CLIP_MEAN) / CLIP_STD)
    return torch.stack(tensors)


def get_model() -> tuple[Detector, torch.device]:
    """Load the detector once, then reuse it for later requests."""
    global _model, _device

    if _model is None:
        with _model_lock:
            if _model is None:
                if not CHECKPOINT.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
                model = Detector(
                    unfreeze_last=checkpoint.get("args", {}).get("unfreeze_last", 0)
                )
                model.load_state_dict(
                    checkpoint["model"], strict=not checkpoint.get("head_only")
                )
                model.to(device).eval()
                _model = model
                _device = device

    return _model, _device


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyse")
def analyse():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose an image first."), 400

    extension = Path(uploaded.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Use a JPG, PNG, WEBP, BMP, or TIFF image."), 400

    try:
        raw = uploaded.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file could not be read as an image."), 400

    try:
        model, device = get_model()
        batch = make_crops(image).to(device)
        precision_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), precision_context:
            score = model(batch).float().sigmoid().mean().item()
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify(error=f"The detector could not run: {exc}"), 500

    percentage = round(score * 100, 1)
    verdict = "Likely AI-generated" if score >= 0.5 else "Likely authentic"
    return jsonify(filename=uploaded.filename, score=percentage, verdict=verdict)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="The image is larger than the 15 MB limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
