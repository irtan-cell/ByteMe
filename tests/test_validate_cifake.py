import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_cifake", ROOT / "scripts/validate_cifake.py")
validate_cifake = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validate_cifake)


def test_validate_reports_all_expected_folders(tmp_path: Path) -> None:
    for split in ("train", "test"):
        for label in ("REAL", "FAKE"):
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(directory / "sample.png")

    report = validate_cifake.validate(tmp_path)

    assert report["total_images"] == 4
    assert report["unreadable_images"] == []
