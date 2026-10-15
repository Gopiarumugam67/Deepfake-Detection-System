"""
Single-image inference: given a raw image (not necessarily pre-cropped),
detect the face, run the classifier, and return a Real/Fake verdict with
a confidence score.

Usage:
    python -m src.inference.image_predictor path/to/image.jpg
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
from src.data.augmentation import get_eval_transforms  # noqa: E402
from src.models.xception_model import load_checkpoint  # noqa: E402
from src.preprocessing.face_extractor import FaceExtractor  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)


class ImagePredictor:
    """Loads a trained checkpoint once and reuses it for repeated predictions."""

    def __init__(self, checkpoint_path: Optional[Path] = None, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = checkpoint_path or (CFG.paths.checkpoints / f"best_{CFG.train.model_name}.pt")

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. Train a model first "
                "(python -m src.training.trainer)."
            )

        self.model = load_checkpoint(checkpoint_path, device=self.device)
        self.face_extractor = FaceExtractor(device=self.device)
        self.transform = get_eval_transforms(CFG.face.image_size)
        logger.info(f"ImagePredictor ready | checkpoint={checkpoint_path.name} | device={self.device}")

    @torch.no_grad()
    def predict(self, image_path: Path) -> dict:
        """
        Predict Real/Fake for a single image.

        Returns:
            {
                "label": "real" | "fake",
                "confidence": float in [0, 1],   # confidence in the predicted label
                "fake_probability": float in [0, 1],
                "face_detected": bool,
            }
        """
        face = self.face_extractor.extract_from_image(Path(image_path))
        if face is None:
            logger.warning(f"No face detected in {image_path}")
            return {
                "label": None,
                "confidence": None,
                "fake_probability": None,
                "face_detected": False,
                "error": "No face detected in image.",
            }

        return self._predict_from_face_array(face)

    @torch.no_grad()
    def _predict_from_face_array(self, face_rgb: np.ndarray) -> dict:
        tensor = self.transform(image=face_rgb)["image"].unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        fake_prob = float(probs[1])
        pred_idx = int(np.argmax(probs))
        label = CFG.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "label": label,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "face_detected": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepfake prediction on a single image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a specific checkpoint.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    predictor = ImagePredictor(checkpoint_path=checkpoint)
    result = predictor.predict(Path(args.image_path))

    if not result["face_detected"]:
        print(f"Result: {result['error']}")
        return

    print(f"Prediction: {result['label'].upper()} (confidence: {result['confidence']:.2%}, "
          f"fake_probability: {result['fake_probability']:.2%})")


if __name__ == "__main__":
    main()
