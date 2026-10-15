"""
Augmentation pipelines.

Deepfake detectors need to be robust to the kinds of degradation that
happen when fake videos get re-compressed and re-uploaded (JPEG artifacts,
blur, slight color shifts) — otherwise the model just learns dataset-specific
GAN fingerprints instead of generalizable cues. The train pipeline reflects
that; val/test stay clean since they're meant to measure true performance.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Augmentation pipeline for training: geometric + compression/noise robustness."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3),
        A.OneOf([
            A.ImageCompression(quality_range=(60, 95), p=1.0),  # re-upload artifacts
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.014, 0.028), p=1.0),
        ], p=0.35),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic pipeline for validation/test/inference: resize + normalize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
