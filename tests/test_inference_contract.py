from pathlib import Path

from aigc_detector.inference import IMAGE_EXTENSIONS, list_images


def test_list_images_is_recursive_and_extension_limited(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "one.JPG").touch()
    (tmp_path / "nested" / "two.png").touch()
    (tmp_path / "notes.txt").touch()

    assert [path.name for path in list_images(tmp_path)] == ["one.JPG", "two.png"]
    assert ".jpg" in IMAGE_EXTENSIONS
