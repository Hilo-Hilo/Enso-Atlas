#!/usr/bin/env python3
"""
Model Evaluation Script for CLAM on Ovarian Cancer Treatment Response.

Features:
- Comprehensive metrics with bootstrap confidence intervals
- ROC-AUC, PR-AUC, precision, recall, F1, specificity
- Calibration analysis with Expected Calibration Error (ECE)
- Reliability diagrams and precision-recall curves
- Attention visualization on test cases
- Decision threshold analysis

Usage:
    python scripts/evaluate_model.py --model_path outputs/training/best_model.pt --data_dir data/demo
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute metric with bootstrap confidence intervals.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_fn: Function(y_true, y_prob) -> float
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        random_seed: Random seed
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_prob)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            estimate = metric_fn(y_true_boot, y_prob_boot)
            bootstrap_estimates.append(estimate)
        except Exception:
            continue
    
    if len(bootstrap_estimates) < 10:
        logger.warning("Too few valid bootstrap samples")
        return point_estimate, point_estimate, point_estimate
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def compute_all_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000
) -> Dict:
    """
    Compute all clinical metrics with bootstrap confidence intervals.
    
    Metrics include:
    - ROC-AUC: Area under ROC curve
    - PR-AUC: Area under Precision-Recall curve (important for imbalanced data)
    - Precision: TP / (TP + FP)
    - Recall (Sensitivity): TP / (TP + FN)
    - Specificity: TN / (TN + FP)
    - F1 Score: Harmonic mean of precision and recall
    - PPV/NPV: Positive/Negative predictive values
    - Brier Score: Mean squared error of predictions
    - Accuracy: Overall correct classification rate
    """
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss,
        confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score,
        average_precision_score
    )
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    results = {}
    
    # Define metric functions
    def auc_roc_fn(y_t, y_p):
        return roc_auc_score(y_t, y_p)
    
    def auc_pr_fn(y_t, y_p):
        """Precision-Recall AUC - better for imbalanced datasets"""
        return average_precision_score(y_t, y_p)
    
    def precision_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return precision_score(y_t, y_pred, zero_division=0)
    
    def recall_fn(y_t, y_p):
        """Also known as sensitivity or true positive rate"""
        y_pred = (y_p >= threshold).astype(int)
        return recall_score(y_t, y_pred, zero_division=0)
    
    def specificity_fn(y_t, y_p):
        """True negative rate"""
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def f1_fn(y_t, y_p):
        """Harmonic mean of precision and recall"""
        y_pred = (y_p >= threshold).astype(int)
        return f1_score(y_t, y_pred, zero_division=0)
    
    def ppv_fn(y_t, y_p):
        """Positive predictive value (same as precision)"""
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def npv_fn(y_t, y_p):
        """Negative predictive value"""
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def brier_fn(y_t, y_p):
        """Brier score - lower is better"""
        return brier_score_loss(y_t, y_p)
    
    def accuracy_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return accuracy_score(y_t, y_pred)
    
    # Compute metrics in clinical priority order
    metrics_to_compute = [
        ("auc_roc", auc_roc_fn, "ROC-AUC"),
        ("auc_pr", auc_pr_fn, "PR-AUC"),
        ("precision", precision_fn, "Precision"),
        ("recall", recall_fn, "Recall (Sensitivity)"),
        ("specificity", specificity_fn, "Specificity"),
        ("f1_score", f1_fn, "F1 Score"),
        ("ppv", ppv_fn, "PPV"),
        ("npv", npv_fn, "NPV"),
        ("accuracy", accuracy_fn, "Accuracy"),
        ("brier_score", brier_fn, "Brier Score"),
    ]
    
    for metric_name, metric_fn, display_name in metrics_to_compute:
        point, lower, upper = bootstrap_metric(
            y_true, y_prob, metric_fn, n_bootstrap=n_bootstrap
        )
        results[metric_name] = {
            "value": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_95": f"{point:.3f} [{lower:.3f}, {upper:.3f}]",
            "display_name": display_name
        }
    
    # Confusion matrix (no CI needed)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results["confusion_matrix"] = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": int(tp + tn + fp + fn)
    }
    
    # Class distribution
    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)
    results["class_distribution"] = {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": n_positive / len(y_true) if len(y_true) > 0 else 0.0
    }
    
    return results


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics and data for reliability diagram.
    
    A well-calibrated model has predictions that match the true frequency
    of positive outcomes. For example, among samples with predicted
    probability 0.8, approximately 80% should be actual positives.
    
    Returns:
        Dictionary with:
        - expected_calibration_error (ECE): Weighted average of calibration gaps
        - maximum_calibration_error (MCE): Worst-case calibration gap
        - average_confidence: Mean predicted probability
        - average_accuracy: Mean actual positive rate
        - calibration_gap: Difference between confidence and accuracy
        - bin_data: Per-bin calibration statistics
    """
    from sklearn.calibration import calibration_curve
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute bin-level statistics
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_count = mask.sum()
        
        if bin_count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_gap = abs(bin_acc - bin_conf)
            ece += (bin_count / len(y_prob)) * bin_gap
            
            bin_data.append({
                "bin_index": i,
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "count": int(bin_count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(bin_gap)
            })
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
    
    # Overall statistics
    avg_confidence = float(y_prob.mean())
    avg_accuracy = float(y_true.mean())
    
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "average_confidence": avg_confidence,
        "average_accuracy": avg_accuracy,
        "calibration_gap": abs(avg_confidence - avg_accuracy),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_bins": n_bins,
        "bin_data": bin_data
    }


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Analyze performance at different decision thresholds.
    
    Useful for selecting optimal threshold based on clinical requirements
    (e.g., maximizing sensitivity for screening vs specificity for confirmation).
    """
    from sklearn.metrics import confusion_matrix, f1_score
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    threshold_results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Youden's J statistic (sensitivity + specificity - 1)
        youden_j = sensitivity + specificity - 1
        
        threshold_results.append({
            "threshold": thresh,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "f1_score": float(f1),
            "youden_j": float(youden_j),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        })
    
    # Find optimal thresholds
    f1_scores = [r["f1_score"] for r in threshold_results]
    youden_scores = [r["youden_j"] for r in threshold_results]
    
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_youden_idx = np.argmax(youden_scores)
    
    return {
        "thresholds": threshold_results,
        "optimal_f1_threshold": thresholds[optimal_f1_idx],
        "optimal_youden_threshold": thresholds[optimal_youden_idx],
        "default_threshold": 0.5
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 100
):
    """
    Plot ROC curve with bootstrap confidence band.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Base ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap for confidence band
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_prob[indices])
        tprs.append(np.interp(mean_fpr, fpr_boot, tpr_boot))
        aucs.append(auc(fpr_boot, tpr_boot))
    
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance (AUC = 0.5)')
    ax.plot(fpr, tpr, color='#2563eb', lw=2, 
            label=f'Model (AUC = {roc_auc:.3f})')
    
    # Confidence band
    ax.fill_between(mean_fpr, 
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    color='#2563eb', alpha=0.2, label='95% CI')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve - Treatment Response Prediction', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for operating point at threshold=0.5
    idx_50 = np.argmin(np.abs(np.array([0.5]) - y_prob.mean()))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curve to {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 100
):
    """
    Plot Precision-Recall curve with bootstrap confidence band.
    
    PR curves are particularly useful for imbalanced datasets as they
    focus on the positive class performance.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Base PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    # Baseline (prevalence)
    baseline = y_true.mean()
    
    # Bootstrap for confidence band
    mean_recall = np.linspace(0, 1, 100)
    precisions_boot = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        prec_boot, rec_boot, _ = precision_recall_curve(
            y_true[indices], y_prob[indices]
        )
        # Interpolate (note: PR curves go from high to low recall)
        prec_interp = np.interp(mean_recall, rec_boot[::-1], prec_boot[::-1])
        precisions_boot.append(prec_interp)
    
    precisions_boot = np.array(precisions_boot)
    mean_prec = precisions_boot.mean(axis=0)
    std_prec = precisions_boot.std(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.axhline(y=baseline, linestyle='--', color='gray', 
               label=f'Baseline (prevalence = {baseline:.2f})')
    ax.plot(recall, precision, color='#059669', lw=2,
            label=f'Model (AP = {ap_score:.3f})')
    
    # Confidence band
    ax.fill_between(mean_recall,
                    np.maximum(mean_prec - 1.96 * std_prec, 0),
                    np.minimum(mean_prec + 1.96 * std_prec, 1),
                    color='#059669', alpha=0.2, label='95% CI')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.set_title('Precision-Recall Curve - Treatment Response Prediction', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR curve to {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bins: int = 10
):
    """
    Plot reliability diagram (calibration curve) with histogram.
    
    A well-calibrated model should have points close to the diagonal.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', 
             label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, 'o-', color='#dc2626', 
             label='Model', markersize=8, linewidth=2)
    
    # Fill calibration gaps
    for i in range(len(prob_pred)):
        if prob_true[i] > prob_pred[i]:
            ax1.fill_between([prob_pred[i], prob_pred[i]], 
                            [prob_pred[i]], [prob_true[i]],
                            color='#dc2626', alpha=0.3)
        else:
            ax1.fill_between([prob_pred[i], prob_pred[i]], 
                            [prob_true[i]], [prob_pred[i]],
                            color='#2563eb', alpha=0.3)
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), 
             color='#6366f1', alpha=0.7, edgecolor='white')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration curve to {output_path}")


def plot_threshold_analysis(
    threshold_data: Dict,
    output_path: Path
):
    """
    Plot performance metrics across different decision thresholds.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    thresholds = [t["threshold"] for t in threshold_data["thresholds"]]
    sensitivity = [t["sensitivity"] for t in threshold_data["thresholds"]]
    specificity = [t["specificity"] for t in threshold_data["thresholds"]]
    f1_scores = [t["f1_score"] for t in threshold_data["thresholds"]]
    ppv = [t["ppv"] for t in threshold_data["thresholds"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, sensitivity, 'o-', color='#dc2626', 
            label='Sensitivity (Recall)', linewidth=2, markersize=6)
    ax.plot(thresholds, specificity, 's-', color='#2563eb', 
            label='Specificity', linewidth=2, markersize=6)
    ax.plot(thresholds, f1_scores, '^-', color='#059669', 
            label='F1 Score', linewidth=2, markersize=6)
    ax.plot(thresholds, ppv, 'd-', color='#7c3aed', 
            label='PPV (Precision)', linewidth=2, markersize=6)
    
    # Mark optimal thresholds
    opt_f1 = threshold_data["optimal_f1_threshold"]
    opt_youden = threshold_data["optimal_youden_threshold"]
    
    ax.axvline(x=opt_f1, linestyle='--', color='#059669', alpha=0.7,
               label=f'Optimal F1 ({opt_f1:.1f})')
    ax.axvline(x=0.5, linestyle=':', color='gray', alpha=0.7,
               label='Default (0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Performance vs Decision Threshold', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved threshold analysis to {output_path}")


def plot_attention_visualization(
    embeddings: np.ndarray,
    attention_weights: np.ndarray,
    coords: Optional[np.ndarray],
    slide_id: str,
    label: int,
    prediction: float,
    output_path: Path
):
    """
    Visualize attention weights for a slide.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Attention distribution
    ax1 = axes[0]
    sorted_indices = np.argsort(attention_weights)[::-1]
    ax1.bar(range(len(attention_weights)), attention_weights[sorted_indices], 
            color='#2563eb', alpha=0.7)
    ax1.set_xlabel('Patch (sorted by attention)', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('Attention Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Right: Spatial heatmap (if coords available)
    ax2 = axes[1]
    if coords is not None and len(coords) == len(attention_weights):
        norm = Normalize(vmin=attention_weights.min(), vmax=attention_weights.max())
        
        scatter = ax2.scatter(
            coords[:, 0], coords[:, 1],
            c=attention_weights,
            cmap='jet',
            s=50,
            alpha=0.8
        )
        plt.colorbar(scatter, ax=ax2, label='Attention')
        ax2.set_xlabel('X coordinate', fontsize=12)
        ax2.set_ylabel('Y coordinate', fontsize=12)
        ax2.set_title('Spatial Attention Map', fontsize=14)
        ax2.invert_yaxis()  # Match image coordinate system
    else:
        # Show histogram instead
        ax2.hist(attention_weights, bins=20, color='#2563eb', alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Attention Weight', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Attention Weight Distribution', fontsize=14)
    
    # Overall title
    label_str = "Responder" if label == 1 else "Non-Responder"
    pred_str = "Responder" if prediction >= 0.5 else "Non-Responder"
    correct = (label == 1 and prediction >= 0.5) or (label == 0 and prediction < 0.5)
    status = "Correct" if correct else "Incorrect"
    
    fig.suptitle(
        f'Slide: {slide_id}\n'
        f'True: {label_str} | Predicted: {pred_str} (p={prediction:.3f}) | {status}',
        fontsize=14, y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attention visualization to {output_path}")


def load_model(model_path: Path):
    """Load trained CLAM model."""
    import torch
    import torch.nn as nn
    
    # Import training config and model
    from train_clam_production import CLAMProductionModel, TrainingConfig
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create config from saved config
    config_dict = checkpoint.get("config", {})
    config = TrainingConfig(**{k: v for k, v in config_dict.items() if hasattr(TrainingConfig, k)})
    
    model = CLAMProductionModel(config)
    model._setup_device()
    model._build_model()
    model._model.load_state_dict(checkpoint["model_state_dict"])
    model._model.to(model._device)
    model._model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    return model


def load_data(data_dir: Path, labels_file: str = "labels.csv"):
    """Load embeddings and labels."""
    labels_path = data_dir / labels_file
    labels_df = pd.read_csv(labels_path)
    
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    
    embeddings_dir = data_dir / "embeddings"
    embeddings_dict = {}
    coords_dict = {}
    
    for _, row in labels_df.iterrows():
        slide_id = row[id_col]
        
        possible_names = [f"{slide_id}.npy", f"demo_{slide_id}.npy"]
        
        for name in possible_names:
            emb_path = embeddings_dir / name
            if emb_path.exists():
                embeddings_dict[slide_id] = np.load(emb_path)
                
                # Try to load coordinates
                coord_name = name.replace(".npy", "_coords.npy")
                coord_path = embeddings_dir / coord_name
                if coord_path.exists():
                    coords_dict[slide_id] = np.load(coord_path)
                break
    
    return embeddings_dict, coords_dict, labels_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLAM Model with Comprehensive Metrics"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/demo",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/evaluation",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Number of bootstrap samples for CI"
    )
    parser.add_argument(
        "--n_attention_samples", type=int, default=5,
        help="Number of samples for attention visualization"
    )
    parser.add_argument(
        "--test_ids_file", type=str, default=None,
        help="Optional JSON file with test IDs (from training)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CLAM Model Evaluation - Comprehensive Analysis")
    logger.info("="*60)
    
    # Load model
    model_path = Path(args.model_path)
    model = load_model(model_path)
    
    # Load data
    data_dir = Path(args.data_dir)
    embeddings_dict, coords_dict, labels_df = load_data(data_dir)
    
    # Determine which samples to evaluate
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    label_col = "label" if "label" in labels_df.columns else "treatment_response"
    
    if args.test_ids_file:
        with open(args.test_ids_file) as f:
            test_data = json.load(f)
            test_ids = test_data.get("test_ids", [])
    else:
        # Use all available data
        test_ids = [sid for sid in labels_df[id_col] if sid in embeddings_dict]
    
    logger.info(f"Evaluating on {len(test_ids)} samples")
    
    # Get predictions for all test samples
    y_true = []
    y_prob = []
    attentions = []
    
    for slide_id in test_ids:
        if slide_id not in embeddings_dict:
            continue
            
        embeddings = embeddings_dict[slide_id]
        prob, attn = model.predict(embeddings)
        
        row = labels_df[labels_df[id_col] == slide_id].iloc[0]
        label = row[label_col]
        if isinstance(label, str):
            label = 1 if label == "responder" else 0
        
        y_true.append(int(label))
        y_prob.append(prob)
        attentions.append((slide_id, attn))
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Compute all metrics with bootstrap CI
    logger.info("\nComputing metrics with bootstrap confidence intervals...")
    metrics = compute_all_metrics_with_ci(
        y_true, y_prob,
        n_bootstrap=args.n_bootstrap
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS (with 95% CI)")
    logger.info("="*60)
    
    logger.info("\nClassification Metrics:")
    for metric_name in ["auc_roc", "auc_pr", "precision", "recall", "specificity", 
                        "f1_score", "ppv", "npv", "accuracy", "brier_score"]:
        if metric_name in metrics:
            display = metrics[metric_name].get("display_name", metric_name)
            ci_str = metrics[metric_name]["ci_95"]
            logger.info(f"  {display}: {ci_str}")
    
    # Confusion matrix
    cm = metrics["confusion_matrix"]
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}")
    logger.info(f"  Total: {cm['total']}")
    
    # Class distribution
    dist = metrics["class_distribution"]
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Positive (Responders): {dist['n_positive']}")
    logger.info(f"  Negative (Non-responders): {dist['n_negative']}")
    logger.info(f"  Prevalence: {dist['prevalence']:.3f}")
    
    # Calibration analysis
    logger.info("\nComputing calibration metrics...")
    calibration = compute_calibration(y_true, y_prob)
    logger.info(f"  Expected Calibration Error (ECE): {calibration['expected_calibration_error']:.4f}")
    logger.info(f"  Maximum Calibration Error (MCE): {calibration['maximum_calibration_error']:.4f}")
    logger.info(f"  Average Confidence: {calibration['average_confidence']:.4f}")
    logger.info(f"  Calibration Gap: {calibration['calibration_gap']:.4f}")
    
    metrics["calibration"] = calibration
    
    # Threshold analysis
    logger.info("\nComputing threshold analysis...")
    threshold_data = compute_threshold_analysis(y_true, y_prob)
    logger.info(f"  Optimal F1 Threshold: {threshold_data['optimal_f1_threshold']:.2f}")
    logger.info(f"  Optimal Youden Threshold: {threshold_data['optimal_youden_threshold']:.2f}")
    
    metrics["threshold_analysis"] = threshold_data
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    
    # ROC curve
    plot_roc_curve(
        y_true, y_prob,
        output_dir / "roc_curve.png",
        n_bootstrap=100
    )
    
    # Precision-Recall curve
    plot_precision_recall_curve(
        y_true, y_prob,
        output_dir / "pr_curve.png",
        n_bootstrap=100
    )
    
    # Calibration curve
    plot_calibration_curve(
        y_true, y_prob,
        output_dir / "calibration_curve.png"
    )
    
    # Threshold analysis plot
    plot_threshold_analysis(
        threshold_data,
        output_dir / "threshold_analysis.png"
    )
    
    # Attention visualizations
    attention_dir = output_dir / "attention_visualizations"
    attention_dir.mkdir(exist_ok=True)
    
    # Select samples for visualization (mix of correct/incorrect, responders/non-responders)
    n_viz = min(args.n_attention_samples, len(attentions))
    
    for i, (slide_id, attn) in enumerate(attentions[:n_viz]):
        coords = coords_dict.get(slide_id)
        
        # Get label and prediction
        row = labels_df[labels_df[id_col] == slide_id].iloc[0]
        label = row[label_col]
        if isinstance(label, str):
            label = 1 if label == "responder" else 0
        
        idx = list(test_ids).index(slide_id)
        prediction = y_prob[idx]
        
        plot_attention_visualization(
            embeddings_dict[slide_id],
            attn,
            coords,
            slide_id,
            label,
            prediction,
            attention_dir / f"attention_{slide_id}.png"
        )
    
    # Save results to JSON
    results = {
        "model_path": str(model_path),
        "n_samples": len(test_ids),
        "n_bootstrap": args.n_bootstrap,
        "metrics": {
            k: v if isinstance(v, dict) else v
            for k, v in metrics.items()
        },
        "predictions": {
            "slide_ids": test_ids,
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist()
        }
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Plots saved to: {output_dir}")
    
    # Print summary table for paper
    logger.info("\n" + "="*60)
    logger.info("SUMMARY TABLE (for paper/submission)")
    logger.info("="*60)
    logger.info("\n| Metric | Value (95% CI) |")
    logger.info("|--------|----------------|")
    
    for metric_name in ["auc_roc", "auc_pr", "precision", "recall", "specificity", 
                        "f1_score", "accuracy", "brier_score"]:
        if metric_name in metrics:
            ci_str = metrics[metric_name]["ci_95"]
            display_name = metrics[metric_name].get("display_name", metric_name.replace("_", " ").title())
            logger.info(f"| {display_name} | {ci_str} |")
    
    # Clinical interpretation
    logger.info("\n" + "="*60)
    logger.info("CLINICAL INTERPRETATION")
    logger.info("="*60)
    
    auc = metrics["auc_roc"]["value"]
    if auc >= 0.9:
        interp = "Excellent discrimination"
    elif auc >= 0.8:
        interp = "Good discrimination"
    elif auc >= 0.7:
        interp = "Acceptable discrimination"
    else:
        interp = "Poor discrimination - model may not be suitable for clinical use"
    
    logger.info(f"\nROC-AUC ({auc:.3f}): {interp}")
    
    ece = calibration["expected_calibration_error"]
    if ece < 0.05:
        cal_interp = "Well calibrated"
    elif ece < 0.1:
        cal_interp = "Reasonably calibrated"
    else:
        cal_interp = "Poorly calibrated - probabilities should be interpreted with caution"
    
    logger.info(f"ECE ({ece:.4f}): {cal_interp}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
