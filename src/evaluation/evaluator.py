"""
Evaluation module.

Runs a trained checkpoint against the held-out test set and produces:
  - accuracy, precision, recall, F1 (via sklearn's classification_report)
  - ROC-AUC
  - confusion matrix plot (PNG)
  - ROC curve plot (PNG)
  - a metrics.json summary for easy reference in the README/resume

Usage:
    python -m src.evaluation.evaluator
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CFG  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.xception_model import load_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_dir=CFG.paths.logs)

REPORTS_DIR = CFG.paths.root / "reports"


@torch.no_grad()
def get_predictions(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

    Returns:
        y_true: ground-truth labels (N,)
        y_pred: predicted class indices (N,)
        y_prob: predicted probability of the "fake" class (N,) — used for ROC-AUC
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())  # index 1 = "fake"

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Save a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CFG.class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> float:
    """Save an ROC curve plot and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curve saved to {out_path} (AUC={auc:.4f})")
    return auc


def evaluate(checkpoint_path: Path | None = None) -> dict:
    """
    Full evaluation pipeline: load checkpoint, run test set, compute metrics,
    save plots + metrics.json.

    Args:
        checkpoint_path: Defaults to checkpoints/best_<model_name>.pt.

    Returns:
        Dict of computed metrics (also written to reports/metrics.json).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = checkpoint_path or (CFG.paths.checkpoints / f"best_{CFG.train.model_name}.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run src/training/trainer.py first."
        )

    model = load_checkpoint(checkpoint_path, device=device)
    loaders = get_dataloaders()

    y_true, y_pred, y_prob = get_predictions(model, loaders["test"], device)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    auc = plot_roc_curve(y_true, y_prob, REPORTS_DIR / "roc_curve.png")
    plot_confusion_matrix(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": auc,
        "num_test_samples": int(len(y_true)),
        "checkpoint": str(checkpoint_path),
        "model_name": CFG.train.model_name,
    }

    report_text = classification_report(y_true, y_pred, target_names=CFG.class_names)
    logger.info(f"\n{report_text}")
    logger.info(f"Metrics: {metrics}")

    (REPORTS_DIR / "classification_report.txt").write_text(report_text)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Full metrics + plots saved to {REPORTS_DIR}")

    return metrics


if __name__ == "__main__":
    evaluate()
