"""
PyTorch Dataset reading the split manifests produced by
src/preprocessing/dataset_builder.py, plus a DataLoader factory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.data.augmentation import get_eval_transforms, get_train_transforms  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)

LABEL_TO_IDX = {name: idx for idx, name in enumerate(CFG.class_names)}  # {"real": 0, "fake": 1}


class DeepfakeFaceDataset(Dataset):
    """Loads (image, label) pairs from a split manifest CSV."""

    def __init__(self, manifest_csv: Path, transform: Optional[Callable] = None) -> None:
        self.df: pd.DataFrame = pd.read_csv(manifest_csv)
        if not {"filepath", "label"}.issubset(self.df.columns):
            raise ValueError(f"Manifest {manifest_csv} must have 'filepath' and 'label' columns")

        # Drop rows whose files no longer exist (e.g. manifest is stale).
        exists_mask = self.df["filepath"].apply(lambda p: Path(p).exists())
        missing = (~exists_mask).sum()
        if missing:
            logger.warning(f"{missing} files listed in {manifest_csv.name} are missing on disk; skipping them")
        self.df = self.df[exists_mask].reset_index(drop=True)

        self.transform = transform
        logger.info(f"Loaded {len(self.df)} samples from {manifest_csv.name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(row["filepath"])
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {row['filepath']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label = LABEL_TO_IDX[row["label"]]

        if self.transform is not None:
            img_rgb = self.transform(image=img_rgb)["image"]
        else:
            img_rgb = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float() / 255.0

        return img_rgb, label


def get_dataloaders(
    image_size: int = CFG.face.image_size,
    batch_size: int = CFG.train.batch_size,
    num_workers: int = CFG.train.num_workers,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from the manifests in data/splits/."""
    loaders = {}
    for split in ("train", "val", "test"):
        manifest = CFG.paths.data_splits / f"{split}.csv"
        transform = get_train_transforms(image_size) if split == "train" else get_eval_transforms(image_size)
        dataset = DeepfakeFaceDataset(manifest, transform=transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders


if __name__ == "__main__":
    loaders = get_dataloaders()
    batch_images, batch_labels = next(iter(loaders["train"]))
    logger.info(f"Sample batch: images={tuple(batch_images.shape)}, labels={tuple(batch_labels.shape)}")
