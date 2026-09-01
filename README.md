# ByteMe — Robust AIGC Image Detector

Point it at a folder of images. It writes a JSON file. Each image gets a score
from 0 to 1 for how likely a generative model made it.

Built for TikTok TechJam 2026, Problem Statement 5: robust detection of
AI-generated images under real-world transformations.

## Results

We validated on 2,000 held-out SID_Set images. The set splits evenly between
real and synthetic.

`TPR@1%FPR` counts how many generated images we catch. It uses the threshold
that wrongly flags 1% of real photos. That number matters more than accuracy.
Users lose trust when a platform flags their own photos.

| Condition | AUROC | Acc@0.5 | TPR@1%FPR | AUROC gap |
|---|---|---|---|---|
| clean | 0.9993 | 0.9905 | 0.9888 | 0.0000 |
| jpeg_90 | 0.9988 | 0.9865 | 0.9746 | 0.0005 |
| jpeg_70 | 0.9985 | 0.9810 | 0.9705 | 0.0008 |
| jpeg_50 | 0.9977 | 0.9775 | 0.9543 | 0.0016 |
| jpeg_30 | 0.9954 | 0.9640 | 0.9299 | 0.0039 |
| blur_0.5 | 0.9993 | 0.9875 | 0.9858 | 0.0000 |
| blur_1.0 | 0.9986 | 0.9825 | 0.9756 | 0.0007 |
| blur_2.0 | 0.9962 | 0.9715 | 0.9004 | 0.0031 |
| resize_0.5 | 0.9984 | 0.9785 | 0.9695 | 0.0009 |
| resize_0.25 | 0.9964 | 0.9685 | 0.9238 | 0.0029 |
| noise_0.02 | 0.9964 | 0.9705 | 0.9045 | 0.0029 |
| noise_0.05 | 0.9916 | 0.9535 | 0.8669 | 0.0077 |
| noise_0.1 | 0.9844 | 0.9330 | 0.7561 | 0.0149 |
| jitter_20 | 0.9991 | 0.9850 | 0.9817 | 0.0002 |
| crop_80 | 0.9993 | 0.9895 | 0.9909 | 0.0000 |

Noise hurts us most. At sigma 0.10 we catch 76% of generated images. On clean
images we catch 99%. AUROC barely moves. The model still ranks images
correctly. It scores every image lower. Our threshold then sits in the wrong
place.

### Out-of-distribution evaluation

In-domain numbers flatter every detector. We ran the model over all 4,998 COCO
val2017 photos. It saw nothing like them during training.

| Threshold | False positive rate |
|---|---|
| 0.5 | 1.66% |
| 0.6109 (picked on SID_Set) | 0.78% |
| 0.9 | 0.12% |

Half the COCO images score below 0.0007. We tuned the 0.6109 threshold to flag
1% of SID_Set photos. It flags 0.78% of COCO photos, so it holds on real images
from a new source.

We then scored the AIGC half: all 8,843 images of **DALL-E 3 Advanced**, the
generated set the organisers specify. Together with COCO that gives us their
full demonstration set, 5,000 real images against 8,843 generated ones, from
sources our training data never covered.

**AUROC on the demonstration set: 0.9574.**

| Threshold | FPR (COCO) | TPR (DALL-E 3) |
|---|---|---|
| 0.5 | 1.66% | 59.1% |
| 0.6109 (picked on SID_Set) | 0.78% | 47.7% |
| 0.9 | 0.12% | 13.9% |

Recalibrating on this set gives 51.7% detection at 1% FPR, at a threshold of
0.5764.

Read those numbers together. AUROC of 0.9574 means the model still ranks
generated images above real ones. Detection near 50% means our threshold sits
too high. The median DALL-E 3 image scores 0.5917, just under the cut, so the
model finds these images suspicious and stops short of committing.

We catch 98.9% of SID_Set synthetics and 47.7% of DALL-E 3 images at the same
threshold. Ranking survives the move to an unseen generator. Calibration does
not. This is the failure mode the problem statement warns about, and it sets
our real ceiling.

## Approach

We freeze a CLIP ViT-B/16 encoder. We train a small MLP head on top of it. The
encoder turns a 224-pixel patch into 768 numbers. The head reads those numbers
and outputs one score. Only 395,777 parameters train. The whole run takes 70
minutes on one GPU.

During training we degrade each image first. We pick one challenge
transformation at random. The head never learns to expect clean input.

We crop each image at its native resolution. We never resize the whole image.
That choice carried the project.

### Native-resolution cropping

SID_Set hands you a shortcut. Every synthetic image measures 1024x1024. Only
3.8% of the real ones do. Resize whole images and the model reads aspect ratio
instead of pixels. It then scores 98% here and fails everywhere else.

A fixed 224 crop hides the image dimensions. It also keeps the pixel statistics
intact. Those statistics carry the generator fingerprints. Resizing smears
them. Our COCO result suggests we avoided the shortcut.

CLIP ViT-B/16 holds about 150M parameters. The limit is 2B.

## Tools and libraries

- **Development:** Python 3.12, Git, tmux over SSH, VS Code
- **Hardware:** AMD Radeon RX 7900 XTX, ROCm 7.1, PyTorch 2.13
- **Model:** `openai/clip-vit-base-patch16` plus our own MLP head, no external API
- **Libraries:** PyTorch, Transformers, pyarrow, scikit-learn, NumPy, Pillow

## Data

We trained on [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set). It
gives us 140,000 training rows and 20,000 validation rows once we drop the
`tampered` class. It ships as parquet shards. We read images from those shards
straight into memory. Saving them back as JPEG would add compression artifacts
and corrupt our signal.

We evaluated against COCO val2017 for real-image false positives.

We left WildFake alone. The organisers built the demo set from it. Training on
it would poison our own benchmark.

We also skipped CIFAKE. The brief lists it, but its images are 32x32. JPEG
quality 30 deletes a 32x32 image. So does blur at sigma 2.0. Neither simulates
a social media re-encode at that size. CLIP also needs 224 pixels, so every
image needs a 7x upscale. That upscale stamps resampling artifacts onto the
training set. Real test images carry no such artifacts.

Raw images, full checkpoints, and prediction dumps stay out of Git. The trained
head weighs 1.6MB and ships in the repo.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
.venv/bin/pip install transformers pyarrow numpy pillow pandas scikit-learn tqdm pyyaml
```

Swap the PyTorch index for the CUDA or CPU one if you do not run ROCm.

## Reproducing results

```bash
# 1. Fetch SID_Set. 126GB.
.venv/bin/hf download saberzl/SID_Set --repo-type dataset --local-dir data/raw/SID_Set

# 2. Build the manifests. Reads metadata columns only, finishes in seconds.
.venv/bin/python scripts/build_sidset_manifest.py

# 3. Train. Roughly 70 minutes on an RX 7900 XTX.
.venv/bin/python scripts/train.py --steps 6000 --workers 12 --batch-size 64

# 4. Evaluate across the transformation grid.
.venv/bin/python scripts/evaluate.py --limit 2000
```

Skip step 3 if you want. `checkpoints/head_only.pt` holds our trained weights.

## Directory inference

```bash
.venv/bin/python scripts/infer.py \
  --input path/to/image_directory \
  --checkpoint checkpoints/head_only.pt \
  --output outputs/predictions.json
```

Each score averages five 224 crops at native resolution. That mirrors how we
trained. You get a JSON array back:

```json
[
  {"image_path": "samples/photo.jpg", "pred": 0.000722},
  {"image_path": "samples/synthetic.png", "pred": 0.995512}
]
```

## Repository layout

```text
data/manifests/          CSV manifests, one row per image
scripts/                 build_sidset_manifest.py, train.py, evaluate.py, infer.py
src/aigc_detector/       transforms.py, dataset.py, model.py
checkpoints/             head_only.pt, our trained MLP head
outputs/                 Metrics tables and prediction dumps
```

## Limitations

**Detection halves on an unseen generator.** We catch 98.9% of SID_Set
synthetics and 47.7% of DALL-E 3 images at the same threshold. Generator
coverage sets our ceiling, not model capacity or training length. Training on
one source teaches one family of fingerprints.

**A single threshold does not travel.** AUROC holds at 0.9574 across the
demonstration set, so the ordering stays sound. The score distribution shifts
under a new generator, and our cut sits above most DALL-E 3 images. Any
deployment needs recalibration per source.

**We ignore tampered images.** SID_Set's `tampered` class holds real
photographs with AI-edited regions. We dropped them from training. Score them
anyway and they land between 0.0001 and 0.62. Partial edits sit outside what
this model decides.

**Calibration breaks before ranking does.** Heavy noise barely moves AUROC. It
shifts our 1% FPR threshold a long way. A deployment needs per-condition
thresholds, or a quality estimate feeding the decision.

**We never tested cross-generator transfer systematically.** One unseen
generator gave us one data point. Holding out generators inside training would
measure the effect properly.

## Future work

1. Train across several generators, holding some out, and measure transfer
   directly rather than inferring it from one unseen set.
2. Add a consistency loss. It pulls clean and degraded versions of one image
   toward the same embedding. That attacks the robustness gap directly.
   Augmentation alone only hopes to cover it.
3. Unfreeze the last two encoder blocks. We have the parameter budget spare.
4. Add a forensic branch reading high-frequency residuals. CLIP's semantic
   features should not carry this alone.

## Team contributions

TODO
