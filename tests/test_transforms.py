from PIL import Image

from aigc_detector.transforms import EVALUATION_CONDITIONS, center_crop, gaussian_noise, jpeg_compression, resize_and_upscale


def test_challenge_transforms_keep_image_dimensions() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    assert jpeg_compression(image, 30).size == image.size
    assert resize_and_upscale(image, 0.25).size == image.size
    assert gaussian_noise(image, 0.1).size == image.size
    assert center_crop(image).size == image.size
    assert set(EVALUATION_CONDITIONS) >= {"clean", "jpeg_q30", "blur_sigma2", "resize_025", "noise_010", "crop_80"}
