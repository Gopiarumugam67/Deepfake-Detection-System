"""
Tests for src/inference/video_predictor.py — focused on the frame
aggregation logic (mean fake-probability -> verdict), since the actual
frame extraction/model inference is covered by conftest mocks, not real
video decoding or a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.video_predictor import VideoPredictor  # noqa: E402


def _make_video_predictor(per_frame_fake_probs: list[float]) -> VideoPredictor:
    """Build a VideoPredictor without running __init__, with a fake image_predictor."""
    predictor = VideoPredictor.__new__(VideoPredictor)
    predictor.frame_stride = 10
    predictor.max_frames = 30

    fake_image_predictor = MagicMock()
    fake_image_predictor.face_extractor.extract_from_video.return_value = [
        np.zeros((224, 224, 3), dtype=np.uint8) for _ in per_frame_fake_probs
    ]

    call_count = {"i": 0}

    def fake_predict_from_face_array(face_rgb: np.ndarray) -> dict:
        p = per_frame_fake_probs[call_count["i"]]
        call_count["i"] += 1
        label = "fake" if p >= 0.5 else "real"
        return {"label": label, "confidence": p if label == "fake" else 1 - p, "fake_probability": p}

    fake_image_predictor._predict_from_face_array = fake_predict_from_face_array
    predictor.image_predictor = fake_image_predictor
    return predictor


class TestVideoPredictor:
    def test_no_faces_detected_returns_error_result(self, tmp_path):
        predictor = _make_video_predictor(per_frame_fake_probs=[])
        predictor.image_predictor.face_extractor.extract_from_video.return_value = []

        result = predictor.predict(tmp_path / "video.mp4")

        assert result["num_frames_with_face"] == 0
        assert result["label"] is None
        assert "error" in result

    def test_mostly_fake_frames_yield_fake_verdict(self, tmp_path):
        predictor = _make_video_predictor(per_frame_fake_probs=[0.9, 0.85, 0.95, 0.6, 0.2])

        result = predictor.predict(tmp_path / "video.mp4")

        assert result["label"] == "fake"
        assert result["num_frames_with_face"] == 5
        assert result["mean_fake_probability"] == pytest.approx(np.mean([0.9, 0.85, 0.95, 0.6, 0.2]))

    def test_mostly_real_frames_yield_real_verdict(self, tmp_path):
        predictor = _make_video_predictor(per_frame_fake_probs=[0.1, 0.05, 0.2, 0.15])

        result = predictor.predict(tmp_path / "video.mp4")

        assert result["label"] == "real"
        assert result["confidence"] == pytest.approx(1 - result["mean_fake_probability"])

    def test_frame_predictions_list_matches_sampled_frame_count(self, tmp_path):
        predictor = _make_video_predictor(per_frame_fake_probs=[0.3, 0.7, 0.4])

        result = predictor.predict(tmp_path / "video.mp4")

        assert len(result["frame_predictions"]) == 3
