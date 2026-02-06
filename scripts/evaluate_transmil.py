#!/usr/bin/env python3
"""
TransMIL Model Evaluation Script.

Loads a trained TransMIL checkpoint and evaluates it on a held-out test set
(or the full dataset if no explicit test split is provided). Generates:

  - Classification report (precision, recall, F1, support)
  - Confusion matrix plot
  - ROC curve plot with bootstrap 95% confidence band
  - Per-slide predictions CSV
  - Summary results JSON

All outputs are saved to the specified results directory.

Usage:
    python scripts/evaluate_transmil.py \
        --checkpoint results/transmil_finetune/best_model.pt \
        --embeddings_dir data/tcga_full/embeddings \
        --labels_file data/tcga_full/labels.csv \
        --output_dir results/transmil_evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

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
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TransMIL, dict]:
    """
    Load a TransMIL model from a checkpoint file.

    Supports two checkpoint formats:
      1. Full checkpoint dict with 'model_state_dict' and 'config' keys
         (produced by train_transmil_finetune.py).
      2. Plain state_dict (produced by older training scripts).

    Returns:
        (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        cfg = {}

    # Defaults matching the standard training config
    model = TransMIL(
        input_dim=cfg.get("input_dim", 384),
        hidden_dim=cfg.get("hidden_dim", 512),
        num_classes=cfg.get("num_classes", 1),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.25),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded TransMIL from %s", checkpoint_path)
    return model, cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load labels CSV (slide_id, label columns required)."""
    df = pd.read_csv(labels_path)
    if "slide_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Labels CSV must have 'slide_id' and 'label' columns.")
    return df


def load_embeddings(
    embeddings_dir: Path, slide_ids: List[str]
) -> Dict[str, np.ndarray]:
    """Load .npy embeddings for the listed slide IDs."""
    data: Dict[str, np.ndarray] = {}
    for sid in slide_ids:
        p = embeddings_dir / f"{sid}.npy"
        if p.exists():
            data[sid] = np.load(p)
    return data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_all(
    model: TransMIL,
    data: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Run inference on all slides.

    Returns dict mapping slide_id -> (probability, attention_weights).
    """
    model.eval()
    results: Dict[str, Tuple[float, np.ndarray]] = {}

    for sid, emb_np in data.items():
        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
        prob, attn = model(emb, return_attention=True)
        results[sid] = (prob.item(), attn.cpu().numpy())

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Non-Responder (0)", "Responder (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix -- TransMIL",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix -> %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 200,
) -> None:
    """Save an ROC curve with bootstrap 95% CI band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Bootstrap for confidence band
    mean_fpr = np.linspace(0, 1, 100)
    tprs: List[np.ndarray] = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_prob[idx])
        tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))

    tprs_arr = np.array(tprs)
    mean_tpr = tprs_arr.mean(axis=0)
    std_tpr = tprs_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance (AUC = 0.50)")
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"TransMIL (AUC = {roc_auc:.3f})")
    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - 1.96 * std_tpr, 0, 1),
        np.clip(mean_tpr + 1.96 * std_tpr, 0, 1),
        color="#2563eb",
        alpha=0.2,
        label="95% CI",
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve -- TransMIL Platinum Sensitivity Prediction")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve -> %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TransMIL model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with per-slide .npy embeddings.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="CSV with slide_id and label columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/transmil_evaluation",
        help="Output directory for plots and results.",
    )
    parser.add_argument(
        "--test_ids",
        type=str,
        default=None,
        help="Optional JSON file listing test slide IDs. If omitted, all slides are used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary prediction.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap iterations for ROC CI.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load model
    model, model_cfg = load_model(Path(args.checkpoint), device)

    # Load data
    labels_df = load_labels(Path(args.labels_file))
    labels_map: Dict[str, int] = dict(
        zip(labels_df["slide_id"], labels_df["label"].astype(int))
    )

    # Determine which slides to evaluate
    if args.test_ids:
        with open(args.test_ids) as f:
            test_ids = json.load(f)
            if isinstance(test_ids, dict):
                test_ids = test_ids.get("test_ids", [])
    else:
        test_ids = list(labels_map.keys())

    embeddings = load_embeddings(Path(args.embeddings_dir), test_ids)
    test_ids = [s for s in test_ids if s in embeddings and s in labels_map]
    logger.info("Evaluating %d slides", len(test_ids))

    if len(test_ids) == 0:
        logger.error("No slides found. Check paths.")
        sys.exit(1)

    # Restrict to test set
    test_data = {s: embeddings[s] for s in test_ids}

    # Inference
    predictions = predict_all(model, test_data, device)

    y_true = np.array([labels_map[s] for s in test_ids])
    y_prob = np.array([predictions[s][0] for s in test_ids])
    y_pred = (y_prob >= args.threshold).astype(int)

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------
    target_names = ["Non-Responder", "Responder"]
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    logger.info("\nClassification Report:\n%s", report_str)

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    # AUC
    try:
        test_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        test_auc = 0.5
    logger.info("AUC-ROC: %.4f", test_auc)

    # Sensitivity / Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    logger.info("Sensitivity: %.4f", sensitivity)
    logger.info("Specificity: %.4f", specificity)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png", n_bootstrap=args.n_bootstrap)

    # ------------------------------------------------------------------
    # Per-slide predictions CSV
    # ------------------------------------------------------------------
    preds_df = pd.DataFrame(
        {
            "slide_id": test_ids,
            "true_label": y_true.tolist(),
            "predicted_prob": y_prob.tolist(),
            "predicted_label": y_pred.tolist(),
        }
    )
    preds_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info("Saved per-slide predictions -> %s", output_dir / "predictions.csv")

    # ------------------------------------------------------------------
    # Results JSON
    # ------------------------------------------------------------------
    results = {
        "checkpoint": str(args.checkpoint),
        "n_slides": len(test_ids),
        "threshold": args.threshold,
        "auc_roc": test_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "classification_report": report_dict,
        "model_config": model_cfg,
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results JSON -> %s", results_path)

    logger.info("\nEvaluation complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
