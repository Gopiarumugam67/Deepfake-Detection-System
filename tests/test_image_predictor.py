"""
Tests for src/inference/image_predictor.py.

The real ImagePredictor.__init__ loads a torch checkpoint and an MTCNN
model — neither is available/desirable in a unit test, so we bypass
__init__ (object.__new__) and inject lightweight fakes for
`face_extractor` and `model`/`transform` instead. This tests the actual
prediction/aggregation *logic*, not torch or MTCNN themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.image_predictor import ImagePredictor  # noqa: E402


def _make_predictor_with_fake_model(fake_probs: list[float]) -> ImagePredictor:
    """
    Build an ImagePredictor without running __init__ (no checkpoint/MTCNN needed),
    wired up so `_predict_from_face_array` returns deterministic fake probabilities
    in sequence across calls.
    """
    predictor = ImagePredictor.__new__(ImagePredictor)
    call_count = {"i": 0}

    def fake_predict_from_face_array(face_rgb: np.ndarray) -> dict:
        p = fake_probs[call_count["i"] % len(fake_probs)]
        call_count["i"] += 1
        label = "fake" if p >= 0.5 else "real"
        return {
            "label": label,
            "confidence": p if label == "fake" else (1 - p),
            "fake_probability": p,
            "face_detected": True,
        }

    predictor._predict_from_face_array = fake_predict_from_face_array
    predictor.face_extractor = MagicMock()
    predictor.transform = MagicMock()
    predictor.model = MagicMock()
    predictor.device = "cpu"
    return predictor


class TestImagePredictor:
    def test_predict_returns_no_face_result_when_extractor_finds_nothing(self, tmp_path):
        predictor = _make_predictor_with_fake_model(fake_probs=[0.9])
        predictor.face_extractor.extract_from_image.return_value = None

        dummy_path = tmp_path / "no_face.jpg"
        dummy_path.touch()
        result = predictor.predict(dummy_path)

        assert result["face_detected"] is False
        assert result["label"] is None
        assert "error" in result

    def test_predict_returns_fake_label_for_high_fake_probability(self, dummy_face_array, tmp_path):
        predictor = _make_predictor_with_fake_model(fake_probs=[0.87])
        predictor.face_extractor.extract_from_image.return_value = dummy_face_array

        dummy_path = tmp_path / "face.jpg"
        dummy_path.touch()
        result = predictor.predict(dummy_path)

        assert result["face_detected"] is True
        assert result["label"] == "fake"
        assert result["fake_probability"] == pytest.approx(0.87)
        assert result["confidence"] == pytest.approx(0.87)

    def test_predict_returns_real_label_for_low_fake_probability(self, dummy_face_array, tmp_path):
        predictor = _make_predictor_with_fake_model(fake_probs=[0.05])
        predictor.face_extractor.extract_from_image.return_value = dummy_face_array

        dummy_path = tmp_path / "face.jpg"
        dummy_path.touch()
        result = predictor.predict(dummy_path)

        assert result["label"] == "real"
        assert result["confidence"] == pytest.approx(0.95)
