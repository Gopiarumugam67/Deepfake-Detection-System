"""
Video inference: sample frames from a video, run face detection + the
classifier on each sampled frame, and aggregate frame-level predictions
into a single video-level verdict.

Usage:
    python -m src.inference.video_predictor path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.inference.image_predictor import ImagePredictor  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)


class VideoPredictor:
    """Aggregates per-frame predictions into a video-level Real/Fake verdict."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        frame_stride: int = 10,
        max_frames: int = 30,
    ) -> None:
        # Reuse ImagePredictor's loaded model + face extractor rather than duplicating setup.
        self.image_predictor = ImagePredictor(checkpoint_path=checkpoint_path, device=device)
        self.frame_stride = frame_stride
        self.max_frames = max_frames

    def predict(self, video_path: Path) -> dict:
        """
        Predict Real/Fake for a video.

        Strategy: sample frames at a fixed stride, run face extraction +
        classification per frame, then aggregate by averaging the fake
        probability across all frames where a face was found. Averaging
        (rather than majority vote) is more robust to a few frames where
        detection or the model is briefly wrong.

        Returns:
            {
                "label": "real" | "fake",
                "confidence": float,
                "mean_fake_probability": float,
                "num_frames_sampled": int,
                "num_frames_with_face": int,
                "frame_predictions": list[dict],   # per-frame detail for debugging/UI
            }
        """
        faces = self.image_predictor.face_extractor.extract_from_video(
            Path(video_path), frame_stride=self.frame_stride, max_frames=self.max_frames
        )

        if not faces:
            logger.warning(f"No faces detected in any sampled frame of {video_path}")
            return {
                "label": None,
                "confidence": None,
                "mean_fake_probability": None,
                "num_frames_sampled": 0,
                "num_frames_with_face": 0,
                "frame_predictions": [],
                "error": "No faces detected in any sampled frame.",
            }

        frame_predictions = []
        fake_probs = []
        for face in faces:
            result = self.image_predictor._predict_from_face_array(face)
            frame_predictions.append(result)
            fake_probs.append(result["fake_probability"])

        mean_fake_prob = float(np.mean(fake_probs))
        label = "fake" if mean_fake_prob >= 0.5 else "real"
        confidence = mean_fake_prob if label == "fake" else (1 - mean_fake_prob)

        logger.info(
            f"{video_path}: {label.upper()} | mean_fake_prob={mean_fake_prob:.4f} | "
            f"frames_with_face={len(faces)}"
        )

        return {
            "label": label,
            "confidence": confidence,
            "mean_fake_probability": mean_fake_prob,
            "num_frames_sampled": len(faces),
            "num_frames_with_face": len(faces),
            "frame_predictions": frame_predictions,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepfake prediction on a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a specific checkpoint.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=30, help="Max number of frames to sample.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    predictor = VideoPredictor(
        checkpoint_path=checkpoint, frame_stride=args.frame_stride, max_frames=args.max_frames
    )
    result = predictor.predict(Path(args.video_path))

    if result["num_frames_with_face"] == 0:
        print(f"Result: {result['error']}")
        return

    print(f"Prediction: {result['label'].upper()} (confidence: {result['confidence']:.2%}, "
          f"mean_fake_probability: {result['mean_fake_probability']:.2%}, "
          f"frames_analyzed: {result['num_frames_with_face']})")


if __name__ == "__main__":
    main()
