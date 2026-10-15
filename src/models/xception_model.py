"""
Model factory: builds a transfer-learning classifier from a pretrained
backbone (Xception / EfficientNet-B0 / ResNet50 / ViT-Base), all via `timm`
so the same interface works regardless of which architecture is selected
in config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)

# Maps our config-friendly names to the exact timm model identifiers.
_TIMM_NAME_MAP = {
    "xception": "xception",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "vit_base": "vit_base_patch16_224",
}


class DeepfakeClassifier(nn.Module):
    """Thin wrapper around a timm backbone with a binary classification head."""

    def __init__(
        self,
        model_name: str = CFG.train.model_name,
        num_classes: int = CFG.train.num_classes,
        pretrained: bool = CFG.train.pretrained,
    ) -> None:
        super().__init__()
        if model_name not in _TIMM_NAME_MAP:
            raise ValueError(
                f"Unknown model_name '{model_name}'. Choose from: {list(_TIMM_NAME_MAP)}"
            )

        timm_name = _TIMM_NAME_MAP[model_name]
        # num_classes set directly on the backbone swaps in a correctly-sized
        # classification head for us instead of us manually replacing `fc`/`classifier`.
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        self.model_name = model_name

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Built {model_name} ({timm_name}) | pretrained={pretrained} | params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(device: str | None = None) -> DeepfakeClassifier:
    """Instantiate the configured model and move it to the appropriate device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeClassifier()
    model.to(device)
    logger.info(f"Model moved to device={device}")
    return model


def save_checkpoint(model: nn.Module, path: Path, extra: dict | None = None) -> None:
    """Save model weights (+ optional metadata like epoch/optimizer state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, device: str | None = None) -> DeepfakeClassifier:
    """Rebuild the configured model architecture and load saved weights into it."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeClassifier()
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded from {path} onto device={device}")
    return model


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, CFG.face.image_size, CFG.face.image_size)
    out = model(dummy.to(next(model.parameters()).device))
    logger.info(f"Sanity check output shape: {tuple(out.shape)}")
