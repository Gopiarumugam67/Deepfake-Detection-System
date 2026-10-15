"""Shared fixtures for the test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.fixture
def tmp_project_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create a throwaway data/{raw,processed,splits} layout for a test run."""
    raw_real = tmp_path / "raw" / "real"
    raw_fake = tmp_path / "raw" / "fake"
    staging = tmp_path / "processed" / "_staging"
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    for d in (raw_real, raw_fake, staging / "real", staging / "fake", processed, splits):
        d.mkdir(parents=True, exist_ok=True)

    yield {
        "root": tmp_path,
        "raw_real": raw_real,
        "raw_fake": raw_fake,
        "staging": staging,
        "processed": processed,
        "splits": splits,
    }
    shutil.rmtree(tmp_path, ignore_errors=True)


def make_dummy_image(path: Path, size: int = 64, color: tuple[int, int, int] = (120, 90, 200)) -> None:
    """Write a small solid-color JPEG to disk — enough for pipeline plumbing tests."""
    if not HAS_CV2:
        pytest.skip("opencv-python not installed")
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def dummy_face_array() -> np.ndarray:
    """A fake RGB face crop the size the model expects (224x224x3)."""
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
