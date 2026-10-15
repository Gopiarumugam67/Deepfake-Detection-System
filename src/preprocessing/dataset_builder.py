"""
Builds train/val/test splits from the extracted face crops and writes:
  1. Physical copies into data/processed/{train,val,test}/{real,fake}/
  2. CSV manifests into data/splits/ (filepath,label) for reproducibility.

Run after face_extractor.py has populated the staging directories.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)


def _collect_labeled_paths(staging_dir: Path) -> pd.DataFrame:
    """Scan staging_dir/{real,fake}/ and build a (filepath, label) DataFrame."""
    records = []
    for class_name in CFG.class_names:
        class_dir = staging_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Missing expected directory: {class_dir}")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"filepath": str(img_path), "label": class_name})

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            f"No labeled images found under {staging_dir}. "
            "Run src/preprocessing/face_extractor.py first."
        )
    return df


def build_splits(staging_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Split extracted faces into stratified train/val/test sets, copy files
    into data/processed/<split>/<class>/, and write manifest CSVs.

    Returns:
        Dict mapping split name -> DataFrame(filepath, label, new_path).
    """
    staging_dir = staging_dir or (CFG.paths.data_processed / "_staging")
    df = _collect_labeled_paths(staging_dir)
    logger.info(f"Found {len(df)} total labeled face crops: "
                f"{df['label'].value_counts().to_dict()}")

    stratify_col = df["label"] if CFG.split.stratify else None
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - CFG.split.train_ratio),
        stratify=stratify_col,
        random_state=CFG.split.random_seed,
    )

    relative_val_ratio = CFG.split.val_ratio / (CFG.split.val_ratio + CFG.split.test_ratio)
    stratify_temp = temp_df["label"] if CFG.split.stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_ratio),
        stratify=stratify_temp,
        random_state=CFG.split.random_seed,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        split_df = split_df.copy()
        new_paths = []
        for _, row in split_df.iterrows():
            src = Path(row["filepath"])
            dst = CFG.paths.data_processed / split_name / row["label"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            new_paths.append(str(dst))
        split_df["filepath"] = new_paths

        manifest_path = CFG.paths.data_splits / f"{split_name}.csv"
        CFG.paths.data_splits.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(manifest_path, index=False)
        logger.info(
            f"{split_name}: {len(split_df)} samples "
            f"{split_df['label'].value_counts().to_dict()} -> {manifest_path}"
        )
        splits[split_name] = split_df

    return splits


if __name__ == "__main__":
    build_splits()
