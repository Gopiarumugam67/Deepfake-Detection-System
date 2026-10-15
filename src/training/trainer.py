"""
Training loop for the Deepfake Media Detection System.

Handles: epoch loop, train/val metrics, early stopping, best-checkpoint
saving, and a per-epoch history CSV so training runs are reproducible
and easy to plot later (loss curves, accuracy curves).

Usage:
    python -m src.training.trainer
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.xception_model import build_model, save_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = CFG.train.early_stopping_patience, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True if val_loss is a new best."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train: bool,
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    Returns:
        (average_loss, accuracy) for the epoch.
    """
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    phase = "train" if train else "val"
    progress = tqdm(loader, desc=phase, leave=False)

    with torch.set_grad_enabled(train):
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(resume_from: Path | None = None) -> Path:
    """
    Full training run: builds model/data/optimizer, loops epochs with
    early stopping, saves the best checkpoint, and writes a history CSV.

    Args:
        resume_from: Optional checkpoint path to resume weights from.

    Returns:
        Path to the best saved checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device={device} | model={CFG.train.model_name} | "
                f"batch_size={CFG.train.batch_size} | epochs={CFG.train.num_epochs}")

    loaders = get_dataloaders()
    model = build_model(device=device)

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Resumed weights from {resume_from}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.train.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=CFG.train.learning_rate, weight_decay=CFG.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping()

    best_ckpt_path = CFG.paths.checkpoints / f"best_{CFG.train.model_name}.pt"
    history: list[dict] = []

    for epoch in range(1, CFG.train.num_epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch}/{CFG.train.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "elapsed_sec": round(elapsed, 1),
        })

        is_best = early_stopping.step(val_loss)
        if is_best:
            save_checkpoint(model, best_ckpt_path, extra={"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})

        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered after epoch {epoch} "
                        f"(no improvement for {CFG.train.early_stopping_patience} epochs)")
            break

    history_path = CFG.paths.logs / f"training_history_{CFG.train.model_name}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Best checkpoint: {best_ckpt_path} (val_loss={early_stopping.best_loss:.4f})")

    return best_ckpt_path


if __name__ == "__main__":
    train_model()
