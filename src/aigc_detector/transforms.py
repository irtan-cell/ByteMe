"""Challenge transformations from TechJam PS5 section 5.2."""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]
JITTER_STRENGTH = 0.2
CROP_FRACTION = 0.8


def jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def noise(img, sigma, rng):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gen = np.random.default_rng(rng.getrandbits(32))
    arr = arr + gen.normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def jitter(img, strength, rng):
    out = img.convert("RGB")
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(1.0 + rng.uniform(-strength, strength))
    return out


def center_crop(img, fraction):
    w, h = img.size
    cw, ch = int(w * fraction), int(h * fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def named(img, name, rng):
    """Apply one transform by its evaluation-grid name, e.g. 'jpeg_30'."""
    if name == "clean":
        return img.convert("RGB")
    kind, _, value = name.partition("_")
    if kind == "jpeg":
        return jpeg(img, int(value))
    if kind == "blur":
        return blur(img, float(value))
    if kind == "resize":
        return resize(img, float(value))
    if kind == "noise":
        return noise(img, float(value), rng)
    if kind == "jitter":
        return jitter(img, JITTER_STRENGTH, rng)
    if kind == "crop":
        return center_crop(img, CROP_FRACTION)
    raise ValueError(f"unknown transform: {name}")


def eval_grid():
    """Every condition the robustness table must report."""
    grid = ["clean"]
    grid += [f"jpeg_{q}" for q in JPEG_QUALITIES]
    grid += [f"blur_{s}" for s in BLUR_SIGMAS]
    grid += [f"resize_{s}" for s in RESIZE_SCALES]
    grid += [f"noise_{s}" for s in NOISE_SIGMAS]
    grid += ["jitter_20", "crop_80"]
    return grid


def sample_train_transform(rng):
    """Pick one condition for training-time augmentation."""
    return rng.choice(eval_grid())
