"""
Central configuration for the Deepfake Media Detection System.

All paths, hyperparameters, and constants live here so that no module
hardcodes magic values. Import this module wherever configuration is needed:

    from config import CFG
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent
    data_raw: Path = root / "data" / "raw"
    data_raw_real: Path = root / "data" / "raw" / "real"
    data_raw_fake: Path = root / "data" / "raw" / "fake"
    data_processed: Path = root / "data" / "processed"
    data_splits: Path = root / "data" / "splits"
    checkpoints: Path = root / "checkpoints"
    logs: Path = root / "logs"


@dataclass(frozen=True)
class FaceExtraction:
    image_size: int = 224          # output crop size (H=W), matches Xception input
    margin: int = 20               # pixels of context added around detected face box
    min_face_confidence: float = 0.90
    device: str = "cuda"           # falls back to "cpu" automatically at runtime


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    stratify: bool = True          # keep real/fake ratio consistent across splits


@dataclass(frozen=True)
class TrainConfig:
    # Default is efficientnet_b0, not xception: on CPU-only hardware (no CUDA),
    # EfficientNet-B0 trains noticeably faster with comparable accuracy.
    # Switch to "xception" here if training on a GPU (e.g. Colab).
    model_name: str = "efficientnet_b0"   # options: xception, efficientnet_b0, resnet50, vit_base
    pretrained: bool = True
    num_classes: int = 2           # 0 = real, 1 = fake
    batch_size: int = 16            # kept small for 8GB RAM / CPU-only training
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 4
    num_workers: int = 2            # 4 cores on i5-1135G7; leave headroom for the OS
    label_smoothing: float = 0.05


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    face: FaceExtraction = field(default_factory=FaceExtraction)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    class_names: tuple = ("real", "fake")


CFG = Config()

if __name__ == "__main__":
    # Quick sanity check: ensure all configured directories exist.
    for name, path in vars(CFG.paths).items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] {name}: {path}")
