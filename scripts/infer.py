#!/usr/bin/env python3
"""Export AIGC probability predictions for every image under a directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_detector.inference import list_images, load_detector, predict_image  # noqa: E402
from aigc_detector.training import best_available_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error("--input must be an existing image directory")
    device = best_available_device()
    model, processor = load_detector(args.checkpoint, device)
    predictions = []
    for image_path in list_images(args.input):
        try:
            predictions.append({"image_path": str(image_path), "pred": round(predict_image(model, processor, image_path, device), 6)})
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable image {image_path}: {error}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
