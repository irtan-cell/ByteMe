#!/usr/bin/env python3
"""Report clean and transformed test-set metrics for a trained detector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_detector.data import ManifestImageDataset  # noqa: E402
from aigc_detector.inference import load_detector  # noqa: E402
from aigc_detector.training import evaluate  # noqa: E402
from aigc_detector.transforms import EVALUATION_CONDITIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_detector(ROOT / config["output"]["checkpoint"], device)
    manifest = config["data"][f"{args.split}_manifest"]
    destination = ROOT / config["output"]["evaluation_dir"]
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, transform in EVALUATION_CONDITIONS.items():
        dataset = ManifestImageDataset(manifest, processor, None if name == "clean" else transform)
        metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=config["batch_size"], shuffle=False), device)
        summaries.append({"condition": name, **metrics})
        pd.DataFrame(predictions).to_csv(destination / f"{args.split}_{name}_predictions.csv", index=False)
    (destination / f"{args.split}_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
