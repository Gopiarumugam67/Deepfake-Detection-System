
"""
Face extraction from images and video frames.

Uses MTCNN (via facenet-pytorch) to detect the dominant face in each
image/frame, crops it with a configurable margin, resizes to a fixed
square size, and writes it to disk.
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
    """Detects and crops the primary face from images and video frames."""

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
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            select_largest=True,
            post_process=False,
            device=self.device,
        )
        logger.info("FaceExtractor initialized on device=%s", self.device)

    def extract_from_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Return a cropped RGB face, or None if reading/detection fails."""
        img_bgr = cv2.imread(str(image_path))

        if img_bgr is None:
            logger.warning("Could not read image: %s", image_path)
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._extract_from_array(img_rgb)

    def _extract_from_array(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            boxes, probs = self.detector.detect(img_rgb)
        except Exception as exc:  # noqa: BLE001
            logger.error("Face detection failed: %s", exc)
            return None

        if boxes is None or probs is None or len(boxes) == 0:
            return None

        best_idx = int(np.argmax(probs))
        if probs[best_idx] < self.min_confidence:
            return None

        face_tensor = self.detector.extract(
            img_rgb, boxes[best_idx : best_idx + 1], None
        )
        if face_tensor is None:
            return None

        return (
            face_tensor.squeeze(0)
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

    def extract_from_video(
        self,
        video_path: Path,
        frame_stride: int = 10,
        max_frames: int = 30,
    ) -> list[np.ndarray]:
        """
        Sample a video and return detected RGB face crops.

        An empty list means no face crops were extracted. A warning is logged
        when the video cannot be opened or contains no readable frames.
        """
        if frame_stride < 1 or max_frames < 1:
            raise ValueError("frame_stride and max_frames must be at least 1.")

        video_path = Path(video_path)

        if not video_path.is_file() or video_path.stat().st_size == 0:
            logger.warning("Video file is missing or empty: %s", video_path)
            return []

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.warning(
                "Invalid or unsupported video; OpenCV could not open: %s",
                video_path,
            )
            cap.release()
            return []

        faces: list[np.ndarray] = []
        frame_idx = 0
        frames_read = 0

        try:
            while len(faces) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_read += 1

                if frame_idx % frame_stride == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = self._extract_from_array(frame_rgb)

                    if face is not None:
                        faces.append(face)

                frame_idx += 1
        finally:
            cap.release()

        if frames_read == 0:
            logger.warning(
                "Video opened but no readable frames were found: %s",
                video_path,
            )
        else:
            logger.info(
                "Read %d frames and extracted %d face crops from %s",
                frames_read,
                len(faces),
                video_path.name,
            )

        return faces

    def save_face(self, face_rgb: np.ndarray, out_path: Path) -> None:
        """Write a cropped RGB face array to disk as an image."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        if not cv2.imwrite(str(out_path), face_bgr):
            logger.error("Could not save extracted face to: %s", out_path)


def batch_extract_directory(
    src_dir: Path,
    dst_dir: Path,
    extractor: Optional[FaceExtractor] = None,
) -> tuple[int, int]:
    """Extract faces from images in a directory, preserving filenames."""
    src_dir = Path(src_dir)

    if not src_dir.is_dir():
        logger.warning("Source directory does not exist: %s", src_dir)
        return 0, 0

    extractor = extractor or FaceExtractor()
    image_paths = sorted(
        p
        for p in src_dir.iterdir()
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

    logger.info(
        "%s -> %s: %d succeeded, %d failed",
        src_dir,
        dst_dir,
        success,
        failed,
    )
    return success, failed


if __name__ == "__main__":
    extractor = FaceExtractor()
    batch_extract_directory(
        CFG.paths.data_raw_real,
        CFG.paths.data_processed / "_staging" / "real",
        extractor,
    )
    batch_extract_directory(
        CFG.paths.data_raw_fake,
        CFG.paths.data_processed / "_staging" / "fake",
        extractor,
    )
