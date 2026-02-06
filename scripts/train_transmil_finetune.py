#!/usr/bin/env python3
"""
TransMIL Fine-Tuning Script for TCGA Ovarian Cancer Platinum Sensitivity Prediction.

This script fine-tunes a TransMIL (Transformer-based Multiple Instance Learning) head
on top of frozen Path Foundation embeddings (384-dim) from the TCGA ovarian cancer
dataset. The goal is slide-level binary classification: predicting platinum sensitivity
(0 = non-responder, 1 = responder).

Key features:
  - Stratified 5-fold cross-validation
  - Class-weighted focal loss for imbalanced data
  - Cosine annealing with warm restarts
  - Per-fold metrics: AUC-ROC, accuracy, sensitivity, specificity
  - Training curve plots (loss and AUC over epochs)
  - Best model checkpoint saving
  - Full results JSON logging

Usage:
    python scripts/train_transmil_finetune.py \
        --embeddings_dir data/tcga_full/embeddings \
        --labels_file data/tcga_full/labels.csv \
        --output_dir results/transmil_finetune

    For a quick smoke test (fewer epochs, 2 folds):
    python scripts/train_transmil_finetune.py \
        --embeddings_dir data/tcga_full/embeddings \
        --labels_file data/tcga_full/labels.csv \
        --output_dir results/transmil_finetune \
        --n_folds 2 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Resolve imports -- works from repo root or scripts/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from transmil import TransMIL  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FinetuneConfig:
    """All hyperparameters for a single fine-tuning run."""
    # Data
    input_dim: int = 384
    # Model
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.25
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    patience: int = 20
    # Cross-validation
    n_folds: int = 5
    # Misc
    seed: int = 42


# ---------------------------------------------------------------------------
# Loss: Focal Loss for class-imbalanced binary classification
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) with optional per-sample weighting.

    Reduces the relative loss for well-classified examples and focuses
    training on hard negatives, which is critical for our heavily-skewed
    responder/non-responder split (~90%/10%).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if pos_weight is not None:
            weight = torch.where(target == 1, 1.0, pos_weight)
            loss = loss * weight

        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load the labels CSV.

    Expected columns: slide_id, label (0/1).
    May also contain: patient_id, platinum_status.
    """
    df = pd.read_csv(labels_path)

    required = {"slide_id", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Labels CSV must contain columns {required}, found {set(df.columns)}"
        )

    return df


def load_slide_embeddings(
    embeddings_dir: Path, slide_ids: List[str]
) -> Dict[str, np.ndarray]:
    """
    Load .npy embeddings for the given slide IDs.

    Returns a dict mapping slide_id -> (n_patches, embed_dim) array.
    Slides whose .npy file is missing are silently skipped.
    """
    data: Dict[str, np.ndarray] = {}
    for sid in slide_ids:
        npy_path = embeddings_dir / f"{sid}.npy"
        if npy_path.exists():
            data[sid] = np.load(npy_path)
    return data


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TransMIL,
    data: Dict[str, np.ndarray],
    labels: Dict[str, int],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pos_weight: torch.Tensor,
) -> float:
    """Run a single training epoch over all slides (stochastic / per-slide)."""
    model.train()
    slide_ids = list(data.keys())
    np.random.shuffle(slide_ids)

    total_loss = 0.0
    for sid in slide_ids:
        optimizer.zero_grad()

        emb = torch.tensor(data[sid], dtype=torch.float32).to(device)
        target = torch.tensor([labels[sid]], dtype=torch.float32).to(device)

        pred = model(emb)
        loss = criterion(pred.view(-1), target, pos_weight=pos_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(slide_ids), 1)


@torch.no_grad()
def evaluate(
    model: TransMIL,
    data: Dict[str, np.ndarray],
    labels: Dict[str, int],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a set of slides.

    Returns a dict with keys: auc, accuracy, sensitivity, specificity, loss.
    """
    model.eval()

    preds: List[float] = []
    true: List[int] = []

    for sid, emb_np in data.items():
        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
        pred = model(emb)
        preds.append(pred.item())
        true.append(labels[sid])

    preds_arr = np.array(preds)
    true_arr = np.array(true)
    pred_binary = (preds_arr >= 0.5).astype(int)

    # AUC (guard against single-class folds)
    try:
        auc = roc_auc_score(true_arr, preds_arr)
    except ValueError:
        auc = 0.5

    acc = accuracy_score(true_arr, pred_binary)

    # Sensitivity (recall of positive class) and specificity
    tn, fp, fn, tp = confusion_matrix(
        true_arr, pred_binary, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # BCE loss for monitoring
    loss = F.binary_cross_entropy(
        torch.tensor(preds_arr, dtype=torch.float32),
        torch.tensor(true_arr, dtype=torch.float32),
    ).item()

    return {
        "auc": auc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "loss": loss,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: Dict[str, List[float]],
    fold: int,
    output_dir: Path,
) -> None:
    """Save loss and AUC curves for a single fold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Loss --
    ax1.plot(epochs, history["train_loss"], label="Train", color="#2563eb", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val", color="#dc2626", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold + 1} -- Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- AUC --
    ax2.plot(epochs, history["val_auc"], label="Val AUC", color="#059669", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title(f"Fold {fold + 1} -- Validation AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves_fold{fold + 1}.png", dpi=150)
    plt.close()


def plot_summary(fold_metrics: List[Dict], output_dir: Path) -> None:
    """Bar chart summarising per-fold AUC, accuracy, sensitivity, specificity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fold_metrics)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(["auc", "accuracy", "sensitivity", "specificity"]):
        vals = [fm[metric] for fm in fold_metrics]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Per-Fold Metrics Summary")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n)])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_fold(
    fold: int,
    train_ids: List[str],
    val_ids: List[str],
    all_data: Dict[str, np.ndarray],
    all_labels: Dict[str, int],
    config: FinetuneConfig,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Train and evaluate a single fold.

    Returns (best_metrics, history).
    """
    logger.info(
        "Fold %d/%d  --  train=%d  val=%d",
        fold + 1, config.n_folds, len(train_ids), len(val_ids),
    )

    train_data = {s: all_data[s] for s in train_ids if s in all_data}
    val_data = {s: all_data[s] for s in val_ids if s in all_data}

    # Compute class weight for the training split
    train_labels_list = [all_labels[s] for s in train_data]
    n_pos = sum(train_labels_list)
    n_neg = len(train_labels_list) - n_pos
    pos_weight_val = n_pos / max(n_neg, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    logger.info(
        "  Class distribution: pos=%d neg=%d  pos_weight=%.2f",
        n_pos, n_neg, pos_weight_val,
    )

    # Build model
    model = TransMIL(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=1,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_auc = 0.0
    best_metrics: Dict[str, float] = {}
    best_state = None
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_accuracy": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "lr": [],
    }

    for epoch in range(config.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_data, all_labels, device, optimizer, criterion, pos_weight
        )
        val_metrics = evaluate(model, val_data, all_labels, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "  Epoch %3d | loss=%.4f | val_auc=%.4f | val_acc=%.4f | "
            "sens=%.4f | spec=%.4f | lr=%.2e | %.1fs",
            epoch + 1,
            train_loss,
            val_metrics["auc"],
            val_metrics["accuracy"],
            val_metrics["sensitivity"],
            val_metrics["specificity"],
            current_lr,
            elapsed,
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_metrics["best_epoch"] = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            logger.info("    -> new best AUC: %.4f", best_auc)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    # Save best model for this fold
    if best_state is not None:
        ckpt_path = output_dir / f"best_model_fold{fold + 1}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "config": asdict(config),
                "fold": fold,
                "best_metrics": best_metrics,
            },
            ckpt_path,
        )
        logger.info("  Saved checkpoint -> %s", ckpt_path)

    # Plot curves
    try:
        plot_training_curves(history, fold, output_dir)
    except Exception as exc:
        logger.warning("Could not plot training curves: %s", exc)

    return best_metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune TransMIL on TCGA ovarian cancer embeddings."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with per-slide .npy embedding files.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="CSV with columns: slide_id, label (0/1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/transmil_finetune",
        help="Where to write checkpoints, plots, and results JSON.",
    )
    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build config
    config = FinetuneConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Seed everything
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    labels_df = load_labels(Path(args.labels_file))
    logger.info("Loaded %d label entries from %s", len(labels_df), args.labels_file)

    embeddings_dir = Path(args.embeddings_dir)
    all_slide_ids = labels_df["slide_id"].tolist()
    all_data = load_slide_embeddings(embeddings_dir, all_slide_ids)

    # Keep only slides that have both labels and embeddings
    available_ids = sorted(all_data.keys())
    labels_map: Dict[str, int] = dict(
        zip(labels_df["slide_id"], labels_df["label"].astype(int))
    )
    available_ids = [s for s in available_ids if s in labels_map]
    available_labels = np.array([labels_map[s] for s in available_ids])

    n_pos = int(available_labels.sum())
    n_neg = len(available_labels) - n_pos
    logger.info(
        "Available slides: %d  (pos=%d [%.1f%%], neg=%d [%.1f%%])",
        len(available_ids),
        n_pos,
        100 * n_pos / len(available_ids),
        n_neg,
        100 * n_neg / len(available_ids),
    )

    if len(available_ids) == 0:
        logger.error("No slides found with both embeddings and labels. Exiting.")
        sys.exit(1)

    # Check embedding dimension
    sample = all_data[available_ids[0]]
    actual_dim = sample.shape[1]
    if actual_dim != config.input_dim:
        logger.warning(
            "Embedding dim=%d differs from config.input_dim=%d; overriding.",
            actual_dim,
            config.input_dim,
        )
        config.input_dim = actual_dim

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    available_ids_arr = np.array(available_ids)

    fold_results: List[Dict] = []
    fold_histories: List[Dict] = []

    logger.info(
        "\n" + "=" * 70 + "\n"
        "  TransMIL Fine-Tuning  --  %d-fold CV\n" + "=" * 70,
        config.n_folds,
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(available_ids_arr, available_labels)
    ):
        train_ids = available_ids_arr[train_idx].tolist()
        val_ids = available_ids_arr[val_idx].tolist()

        metrics, history = run_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            all_data=all_data,
            all_labels=labels_map,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        fold_results.append(metrics)
        fold_histories.append(history)

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 70)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity"]
    agg: Dict[str, Dict[str, float]] = {}

    for m in metric_names:
        vals = [fr[m] for fr in fold_results if m in fr]
        agg[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": vals,
        }
        logger.info(
            "  %-15s  %.4f +/- %.4f   %s",
            m,
            agg[m]["mean"],
            agg[m]["std"],
            ["%.4f" % v for v in vals],
        )

    # Pick the fold with highest AUC as the "best model"
    best_fold = int(np.argmax([fr.get("auc", 0) for fr in fold_results]))
    best_ckpt = output_dir / f"best_model_fold{best_fold + 1}.pt"
    overall_best = output_dir / "best_model.pt"

    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, overall_best)
        logger.info(
            "\nBest fold: %d (AUC=%.4f) -- copied to %s",
            best_fold + 1,
            fold_results[best_fold]["auc"],
            overall_best,
        )

    # Summary plot
    try:
        plot_summary(fold_results, output_dir)
    except Exception as exc:
        logger.warning("Could not plot summary: %s", exc)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "device": str(device),
        "n_slides": len(available_ids),
        "class_distribution": {"positive": n_pos, "negative": n_neg},
        "aggregate_metrics": agg,
        "per_fold_metrics": fold_results,
        "per_fold_histories": fold_histories,
        "best_fold": best_fold + 1,
        "best_fold_auc": fold_results[best_fold].get("auc", 0),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", results_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
