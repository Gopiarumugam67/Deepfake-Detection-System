"""
Face extraction from images and video frames.

Uses MTCNN (via facenet-pytorch) to detect the dominant face in each
image/frame, crops it with a configurable margin, resizes to a fixed
square size, and writes it to disk. This is the first stage of the
preprocessing pipeline — every downstream module (augmentation, dataset
splitting, training) consumes the output of this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)


class FaceExtractor:
    """Detects and crops the primary face from an image using MTCNN."""

    def __init__(
        self,
        image_size: int = CFG.face.image_size,
        margin: int = CFG.face.margin,
        min_confidence: float = CFG.face.min_face_confidence,
        device: Optional[str] = None,
    ) -> None:
        self.image_size = image_size
        self.margin = margin
        self.min_confidence = min_confidence
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            select_largest=True,   # if multiple faces, keep the largest (most prominent)
            post_process=False,    # we handle normalization ourselves at training time
            device=self.device,
        )
        logger.info(f"FaceExtractor initialized on device={self.device}")

    def extract_from_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Detect and crop the primary face from a single image file.

        Args:
            image_path: Path to a .jpg/.png image.

        Returns:
            Cropped face as an HxWx3 RGB uint8 numpy array, or None if no
            face was detected above the confidence threshold.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._extract_from_array(img_rgb)

    def _extract_from_array(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            boxes, probs = self.detector.detect(img_rgb)
        except Exception as exc:  # noqa: BLE001 - MTCNN can raise various backend errors
            logger.error(f"Face detection failed: {exc}")
            return None

        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(np.argmax(probs))
        if probs[best_idx] < self.min_confidence:
            return None

        face_tensor = self.detector.extract(img_rgb, boxes[best_idx : best_idx + 1], None)
        if face_tensor is None:
            return None

        face_np = face_tensor.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        return face_np

    def extract_from_video(
        self, video_path: Path, frame_stride: int = 10, max_frames: int = 30
    ) -> list[np.ndarray]:
        """
        Sample frames from a video at a fixed stride and extract a face crop
        from each sampled frame.

        Args:
            video_path: Path to a video file.
            frame_stride: Take every Nth frame.
            max_frames: Stop after collecting this many valid face crops.

        Returns:
            List of cropped face arrays (HxWx3 RGB uint8).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []

        faces: list[np.ndarray] = []
        frame_idx = 0
        while cap.isOpened() and len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self._extract_from_array(frame_rgb)
                if face is not None:
                    faces.append(face)
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(faces)} face crops from {video_path.name}")
        return faces

    def save_face(self, face_rgb: np.ndarray, out_path: Path) -> None:
        """Write a cropped RGB face array to disk as an image."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr)


def batch_extract_directory(
    src_dir: Path, dst_dir: Path, extractor: Optional[FaceExtractor] = None
) -> tuple[int, int]:
    """
    Run face extraction over every image in src_dir and save crops to dst_dir,
    preserving filenames.

    Args:
        src_dir: Directory of raw images (e.g. data/raw/real).
        dst_dir: Directory to write extracted face crops to.
        extractor: Reuse an existing FaceExtractor instance, or create a new one.

    Returns:
        (num_success, num_failed) counts.
    """
    extractor = extractor or FaceExtractor()
    image_paths = sorted(
        p for p in Path(src_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    success, failed = 0, 0
    for img_path in image_paths:
        face = extractor.extract_from_image(img_path)
        if face is None:
            failed += 1
            continue
        extractor.save_face(face, Path(dst_dir) / img_path.name)
        success += 1

    logger.info(f"{src_dir} -> {dst_dir}: {success} succeeded, {failed} failed (no face detected)")
    return success, failed


if __name__ == "__main__":
    # Example run: extract faces for both classes from raw -> processed staging.
    extractor = FaceExtractor()
    batch_extract_directory(CFG.paths.data_raw_real, CFG.paths.data_processed / "_staging" / "real", extractor)
    batch_extract_directory(CFG.paths.data_raw_fake, CFG.paths.data_processed / "_staging" / "fake", extractor)
