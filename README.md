# ByteMe — Robust AIGC Image Detector

Point it at a folder of images and it writes a JSON file containing a score
from 0 to 1 per image, for how likely a generative model made it.

Built for TikTok TechJam 2026, Problem Statement 5: robust detection of
AI-generated images under real-world transformations.

## Results

Validation runs on 2,000 held-out SID_Set images, split evenly between real and
synthetic. Each row applies one condition from the brief's transformation grid
to every image in that set.

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

Noise is where the model struggles. At sigma 0.10, 76% of generated images are
caught, against 99% on clean ones. AUROC only slips from 0.9993 to 0.9844, so
the ordering is intact. This means the scores have drifted below a threshold
set on clean data.

### Out-of-distribution evaluation

We ran the model over all 5,000 COCO val2017 photos, since in-domain results
tend to overstate how well a detector generalises.

| Threshold | False positive rate |
|---|---|
| 0.5 | 1.66% |
| 0.6109 (picked on SID_Set) | 0.78% |
| 0.9 | 0.12% |

Half of COCO scores below 0.0007. We tuned the 0.6109 threshold to flag 1% of
SID_Set photos. Here it flags 0.78%, which means it transfers to real images
from a new source.

For the generated half we used all 8,843 images of DALL-E 3 Advanced. Together
with COCO that forms the demonstration benchmark named in the brief. That gives
us 5,000 real images against 8,843 generated ones, all from sources absent from
our training data.

**AUROC on the demonstration set: 0.9574.**

| Threshold | FPR (COCO) | TPR (DALL-E 3) |
|---|---|---|
| 0.5 | 1.66% | 59.1% |
| 0.6109 (picked on SID_Set) | 0.78% | 47.7% |
| 0.9 | 0.12% | 13.9% |

Recalibrating on this set causes detection to reach 51.7% at 1% FPR, with the
threshold moving down to 0.5764.

Those two numbers say different things. An AUROC of 0.9574 means generated
images still sort above real ones almost every time. Detection near 50% means
our cut lands in the wrong place. The median DALL-E 3 image scores 0.5917,
which sits just under the threshold.

## Approach

A frozen CLIP ViT-B/16 encoder handles feature extraction, turning a 224-pixel
patch into 768 numbers. A small MLP head then reads those numbers and outputs
one score. This let us train on only 395,777 parameters, decreasing training
time to 70 minutes on one GPU.

To keep the head from only ever seeing clean input, each image gets degraded
before it reaches the encoder, using one challenge transformation picked at
random.

CLIP ViT-B/16 holds about 150M parameters, against a limit of 2B.

### Native-resolution cropping

SID_Set contains a shortcut. Every synthetic image measures 1024x1024, while
only 3.8% of the real ones do. A model fed whole resized images can separate
the two classes on aspect ratio alone, which would score well here and
generalise to nothing.

Cropping a fixed 224 patch removes image dimensions from the input entirely. It
also leaves the pixel statistics intact, and generator fingerprints live in
those statistics rather than in the composition. Our COCO false positive rate
is consistent with having avoided the shortcut, though we did not train a
resize-based model to confirm it.

## Tools and libraries

- **Development:** VS Code, Git
- **Hardware:** AMD Radeon RX 7900 XTX, ROCm 7.1
- **Model:** `openai/clip-vit-base-patch16` plus our own MLP head, no external API
- **Libraries:** Python 3.12, PyTorch 2.13, Transformers, pyarrow, scikit-learn, NumPy, Pillow

## Data

Training data comes from [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set),
which gives 140,000 rows for training and 20,000 for validation once the
`tampered` class is dropped. Images are decoded straight into memory and never
written back to disk, since re-encoding them as JPEG would add compression
artifacts and obscure the generator fingerprints the model reads.

Evaluation uses COCO val2017 and DALL-E 3 Advanced.

WildFake stayed out of training entirely, since the demonstration set is drawn
from it. Training on it would have contaminated our own evaluation.

We also skipped CIFAKE, despite the brief listing it. Its images are 32x32, and
at that size JPEG quality 30 removes most of the detail rather than simulating
a social media re-encode. Blur at sigma 2.0 has the same effect. CLIP needs 224
pixels as well, so every image would need a 7x upscale, stamping resampling
artifacts across the training set that no real test image carries.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
```

### CUDA

```bash
.venv/bin/pip install -r requirements.txt
```

### ROCm

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
.venv/bin/pip install -r requirements.txt
```

## Reproducing results

```bash
# 1. Fetch SID_Set. 126GB.
.venv/bin/hf download saberzl/SID_Set --repo-type dataset --local-dir data/raw/SID_Set

# 2. Build the manifests.
.venv/bin/python scripts/build_sidset_manifest.py

# 3. Train. Roughly 70 minutes on an RX 7900 XTX.
.venv/bin/python scripts/train.py --steps 6000 --workers 12 --batch-size 64

# 4. Evaluate across the transformation grid.
.venv/bin/python scripts/evaluate.py --limit 2000
```

Step 3 is optional, since `checkpoints/head_only.pt` already holds our trained
weights.

COCO val2017 downloads from http://images.cocodataset.org/zips/val2017.zip. The
DALL-E 3 Advanced images sit inside WildFake on ModelScope, under
`Images/Diffusion_based/DALLE.zip`, in the `DALLE/Advanced` folder once
unzipped. The demonstration set numbers come from running inference over each
source and comparing the score distributions:

```bash
.venv/bin/python scripts/infer.py --input path/to/coco/val2017 --output outputs/coco_val2017.json
.venv/bin/python scripts/infer.py --input path/to/DALLE/Advanced --output outputs/dalle3_advanced.json
```

## Directory inference

```bash
.venv/bin/python scripts/infer.py \
  --input path/to/image_directory \
  --checkpoint checkpoints/head_only.pt \
  --output outputs/predictions.json
```

Every score averages five 224 crops taken at native resolution, which reduces
the variance from any single crop. The output is a JSON array:

```json
[
  {"image_path": "samples/photo.jpg", "pred": 0.000722},
  {"image_path": "samples/synthetic.png", "pred": 0.995512}
]
```

## Local web interface

The ByteMe interface lets you select an image, preview it, and run the same
five-crop detector from a browser. From the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`. The first analysis may take longer while
the CLIP backbone is loaded. Uploaded images are processed in memory and are
not saved by the app.

## Repository layout

```text
configs/                 Experiment settings
data/manifests/          CSV manifests, one row per image
scripts/                 build_sidset_manifest.py, train.py, evaluate.py, infer.py
src/aigc_detector/       transforms.py, dataset.py, model.py
tests/                   Output format and augmentation checks
checkpoints/             head_only.pt, our trained MLP head
outputs/                 Metrics tables and the prediction dumps behind our numbers
```

The repository excludes raw images and full checkpoints. The trained head is
included, allowing the detector to run without retraining.

## Limitations

Our detection rate halves against a generator absent from training. We catch
98.9% of SID_Set synthetics but only 47.7% of DALL-E 3 images at the same
threshold, which points at generator coverage as the binding constraint rather
than model capacity or training length. A single training source teaches one
family of fingerprints.

The threshold itself does not transfer either. AUROC stays at 0.9574 across the
demonstration set, so the ordering remains correct, but the score distribution
shifts under a new generator and our cut sits above most DALL-E 3 images. The
same pattern appears under heavy noise, where sigma 0.10 barely moves AUROC yet
pushes the 1% FPR threshold a long way. Either case would require recalibration
before deployment, or a quality estimate feeding the decision.

DALL-E 3 shipped in 2023, so our one out-of-domain generator is already dated.
Current models leave different artifacts, and a single held-out generator gives
one data point rather than a transfer curve.

We did not train on tampered images. SID_Set's `tampered` class holds real
photographs with AI-edited regions, which we dropped, and their scores scatter
between 0.0001 and 0.62 when we run them through anyway.

## Future work

The obvious next step is broader generator coverage. GenImage supplies eight,
including Midjourney and two Stable Diffusion versions, and Chameleon supplies
around 26,000 test images scraped from AI art communities, all of which fooled
human annotators. WildRF would test the degradation side, since its images come
from Facebook, Reddit and Twitter rather than a simulated pipeline.

None of those solve the currency problem on their own, because their generators
date from 2022 and 2023. Models released since then, such as Z-Image, Flux and
Qwen-Image, appear in far smaller benchmarks like DiTFake, or in nothing public
at all. We could build our own dataset instead, generating images locally from
current models and pairing them with photographs, which would close the gap
faster than waiting for a public benchmark.

With several generators in hand, we could train on all but one and test on the
held-out one, rotating through each in turn. That produces a transfer curve
rather than the single data point we have now.

Relying on CLIP alone is the deeper problem. It was trained to match images
against captions, so its features describe content, and generator fingerprints
are not content. They sit in high-frequency residuals, frequency spectra and
noise prints, which a second branch could read alongside the semantic features.

Two smaller changes are cheap enough to try immediately. A consistency loss
would pull clean and degraded versions of the same image toward one embedding,
which targets the robustness gap directly. Unfreezing the last two encoder
blocks would let the encoder adapt at all, and 150M parameters against a 2B
limit leaves room for it.

## Team contributions

Irvin set up the project scaffold, the CIFAKE data pipeline, and the first
training and evaluation entry points. He also built the demo interface and
produced the video.

Tristan replaced the data pipeline with SID_Set, wrote the transformation grid,
the streaming dataset and the detector head, ran the training, robustness and
demonstration set evaluations, and wrote this README.

Elijah designed the video thumbnail.

## Error analysis

Our worst false positives are real COCO photographs scoring 0.91 to 0.97, five
of roughly 5,000. At the calibrated threshold of 0.6109 the false positive rate
is 0.78%, so around 39 authentic photos would be flagged.

The false negatives are more revealing. The lowest-scoring DALL-E 3 images
score 0.0000, and four of the five worst share an identical filename across
different folders, which suggests duplicated images inside the WildFake DALL-E
subset. Our 47.7% detection rate is computed over that set as published.

The dominant error is not confident misclassification but indecision. The
median DALL-E 3 image scores 0.5917, just below our threshold of 0.6109. The
model finds most of these images suspicious without committing, which is why
AUROC stays at 0.9574 while detection sits near 50%. A threshold recalibrated
on this source recovers 51.7% at the same 1% false positive rate.
