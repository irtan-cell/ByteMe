# ByteMe — Robust AIGC Image Detector

An image-level detector for TikTok TechJam 2026, Problem Statement 5: **Robust Detection of AI-Generated Images Under Real-World Transformations**. Given a directory of images, it exports an AIGC probability for every valid image as JSON.

## Approach

The first working model is intentionally compact and reproducible:

1. A frozen `openai/clip-vit-base-patch16` image encoder produces a general visual embedding.
2. A small MLP classification head is trained to predict `0` (real) or `1` (AI-generated).
3. Training applies one of the challenge transformations (JPEG compression, blur, downscale/upscale, noise, colour jitter, or centre crop) so the classifier is exposed to post-processed images.
4. Evaluation reports clean and per-transformation metrics, including AUROC and the robustness gap.

CLIP ViT-B/16 is well below the challenge's 2B-parameter limit. Freezing it makes the baseline feasible on hackathon hardware; later experiments can unfreeze the final encoder blocks or add a lightweight forensic branch only if evaluation shows it is worthwhile.

## Tools, models, and libraries

- **Development:** VS Code, Python 3.10+, Git, and a Jupyter notebook for quick error analysis.
- **Model:** Hugging Face `openai/clip-vit-base-patch16` plus a PyTorch MLP head. No external inference API is required.
- **Frameworks:** PyTorch, Hugging Face Transformers, torchvision, scikit-learn, pandas, Pillow, NumPy, PyYAML, and tqdm.

## Data plan

- **Training / in-domain validation:** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). Its real and synthetic labels provide the initial binary training signal.
- **Out-of-distribution evaluation:** [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) and a documented, manageable subset of [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary). These are never mixed into the final held-out test split without being declared.
- **Organiser demonstration validation:** COCO val2017 (real) and DALL·E Advanced (AIGC), exactly as specified in the brief. They are not used during training.

All raw images, model checkpoints, and generated predictions are excluded from Git. Respect each dataset's licence and terms before downloading or redistributing it.

## Repository layout

```text
configs/                 Experiment settings and transformation severities
data/{raw,manifests}/    Downloaded data and deterministic CSV manifests (Git-ignored)
notebooks/               Optional VS Code/Jupyter error analysis
scripts/                 Train, evaluate, and directory-inference entry points
src/aigc_detector/       Reusable data, model, augmentation, and utility code
tests/                   Fast tests for output and augmentation contracts
checkpoints/             Trained model weights (Git-ignored)
outputs/                 Metrics, charts, and JSON predictions (Git-ignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Place CIFAKE under `data/raw/CIFAKE/` using its published `train/REAL`, `train/FAKE`, `test/REAL`, and `test/FAKE` layout. Build deterministic manifests with:

```bash
python scripts/build_cifake_manifest.py
```

Training and clean-versus-transformed evaluation are the next implementation milestones; their intended settings already live in `configs/baseline.yaml`.

## Required directory inference

After training, generate the submission-format predictions with:

```bash
python scripts/infer.py \
  --input path/to/image_directory \
  --checkpoint checkpoints/best.pt \
  --output outputs/predictions.json
```

The output is a JSON array. Every item has an `image_path` and `pred`, where `pred` is the probability from 0 to 1 that the image is AI-generated.

```json
[
  {"image_path": "samples/photo.jpg", "pred": 0.0842},
  {"image_path": "samples/synthetic.png", "pred": 0.9176}
]
```

## Evaluation protocol

The evaluation script compares clean performance against the transformations in the challenge brief: JPEG qualities 90/70/50/30, Gaussian blur (sigma 0.5/1.0/2.0), resize/downscale, Gaussian noise, ±20% colour jitter, and 80% centre crop. It writes aggregate metrics, per-condition metrics, and per-image predictions so false positives and false negatives can be inspected.

## Limitations

This is a decision-support prototype, not proof of image provenance. The baseline can overfit to generators or resolutions represented in the training set, and post-processing or unseen generators may lower confidence reliability. We will explicitly report clean versus transformed performance, per-source metrics, and representative errors.
