"""Real-world redistribution transforms specified in the TechJam brief."""
from __future__ import annotations

import io
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_and_upscale(image: Image.Image, scale: float) -> Image.Image:
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    downscaled = image.resize(size, Image.Resampling.BICUBIC)
    return downscaled.resize(image.size, Image.Resampling.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean noise where sigma is expressed on the [0, 1] pixel scale."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(array + np.random.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8), mode="RGB")


def center_crop(image: Image.Image, fraction: float = 0.8) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = round(width * fraction), round(height * fraction)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


def colour_jitter(image: Image.Image, factor: float) -> Image.Image:
    """Apply one bounded brightness/contrast/saturation adjustment."""
    operation = random.choice((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color))
    return operation(image).enhance(factor)


def random_challenge_transform(image: Image.Image) -> Image.Image:
    """Apply zero or one transformation so labels remain unchanged during training."""
    options: list[Callable[[Image.Image], Image.Image]] = [
        lambda value: jpeg_compression(value, random.choice((90, 70, 50, 30))),
        lambda value: gaussian_blur(value, random.choice((0.5, 1.0, 2.0))),
        lambda value: resize_and_upscale(value, random.choice((0.5, 0.25))),
        lambda value: gaussian_noise(value, random.choice((0.02, 0.05, 0.10))),
        lambda value: colour_jitter(value, random.choice((0.8, 1.2))),
        center_crop,
    ]
    return random.choice(options)(image) if random.random() < 0.7 else image


EVALUATION_CONDITIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda image: image,
    "jpeg_q30": lambda image: jpeg_compression(image, 30),
    "blur_sigma2": lambda image: gaussian_blur(image, 2.0),
    "resize_025": lambda image: resize_and_upscale(image, 0.25),
    "noise_010": lambda image: gaussian_noise(image, 0.10),
    "crop_80": center_crop,
}
