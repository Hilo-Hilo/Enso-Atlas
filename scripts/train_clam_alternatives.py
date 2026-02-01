#!/usr/bin/env python3
"""
Alternative Training Approaches for TCGA Ovarian Cancer.

Addresses class imbalance issues with multiple strategies:
1. Survival-based labels (better balance than platinum response)
2. Random oversampling of minority class
3. Focal loss instead of cross-entropy
4. Optimal threshold selection (Youden's J)

Usage:
    python scripts/train_clam_alternatives.py --label_type survival --oversample
    python scripts/train_clam_alternatives.py --label_type recurrence --focal_loss
    python scripts/train_clam_alternatives.py --label_type platinum --oversample --optimal_threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    input_dim: int = 384
    hidden_dim: int = 256
    attention_heads: int = 1
    dropout: float = 0.25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    random_seed: int = 42
    
    # Alternative training options
    label_type: str = "survival"  # "platinum", "survival", "recurrence"
    use_oversampling: bool = True
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_optimal_threshold: bool = True
    
    # Survival threshold (months)
    survival_threshold: int = 36  # Poor prognosis if OS < this OR deceased


def load_clinical_data(clinical_path: Path) -> pd.DataFrame:
    """Load and preprocess TCGA clinical data."""
    df = pd.read_csv(clinical_path)
    logger.info(f"Loaded clinical data: {len(df)} patients")
    return df


def create_labels(df: pd.DataFrame, label_type: str, survival_threshold: int = 36) -> pd.DataFrame:
    """
    Create binary labels based on different endpoints.
    
    Label types:
    - platinum: platinum_sensitive column (sensitive=1, resistant=0)
    - survival: poor prognosis (OS < threshold OR deceased) = 0, good = 1
    - recurrence: recurred = 0, disease-free = 1
    
    Returns DataFrame with patient_id and label columns.
    """
    df = df.copy()
    
    if label_type == "platinum":
        # Original platinum sensitivity label
        df = df[df["platinum_sensitive"].isin(["sensitive", "resistant"])].copy()
        df["label"] = (df["platinum_sensitive"] == "sensitive").astype(int)
        label_name = "Platinum Response"
        
    elif label_type == "survival":
        # Survival-based label: good prognosis = OS >= threshold AND living
        df = df[df["os_months"].notna() & df["os_status"].notna()].copy()
        df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
        df["deceased"] = df["os_status"].str.contains("DECEASED", na=False)
        
        # Good prognosis: OS >= threshold AND still living
        # Poor prognosis: OS < threshold OR deceased
        df["label"] = ((df["os_months"] >= survival_threshold) & (~df["deceased"])).astype(int)
        label_name = f"Survival (OS>={survival_threshold}mo & living)"
        
    elif label_type == "recurrence":
        # Disease-free survival based label
        df = df[df["dfs_status"].notna()].copy()
        df["recurred"] = df["dfs_status"].str.contains("Recurred", na=False)
        
        # Good outcome: disease-free, Poor outcome: recurred
        df["label"] = (~df["recurred"]).astype(int)
        label_name = "Disease-Free Status"
        
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    
    # Count distribution
    pos = df["label"].sum()
    neg = len(df) - pos
    ratio = max(pos, neg) / min(pos, neg) if min(pos, neg) > 0 else float("inf")
    
    logger.info(f"Label type: {label_name}")
    logger.info(f"Distribution: positive={pos}, negative={neg}, ratio={ratio:.2f}:1")
    
    return df[["patient_id", "label"]].copy()


def load_embeddings_with_labels(
    embeddings_dir: Path,
    labels_df: pd.DataFrame,
    clinical_df: pd.DataFrame
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load embeddings and match with labels.
    
    Supports two label formats:
    1. Clinical data with patient_id (from clinical.csv)
    2. Direct labels with slide_id (from labels.csv: slide_id,patient_id,label,platinum_status)
    
    Returns:
        embeddings_list: List of embedding arrays
        labels: List of binary labels
        patient_ids: List of patient/slide IDs
    """
    embeddings_list = []
    labels = []
    patient_ids = []
    
    # Build label maps (try both patient_id and slide_id based matching)
    patient_label_map = {}
    slide_label_map = {}
    
    if "patient_id" in labels_df.columns:
        patient_label_map = dict(zip(labels_df["patient_id"], labels_df["label"]))
    
    # Check for slide_id column (new format from labels.csv)
    if "slide_id" in labels_df.columns:
        slide_label_map = dict(zip(labels_df["slide_id"], labels_df["label"]))
        logger.info(f"Using slide-level labels: {len(slide_label_map)} slides")
    
    # Scan embedding files
    for emb_file in embeddings_dir.glob("*.npy"):
        if "_coords" in emb_file.name:
            continue
            
        name = emb_file.stem
        
        # Try slide_id match first (more specific)
        # Slide IDs in labels.csv may include UUID suffix: TCGA-13-0794-01A-01-BS1.a4d22c1b-...
        slide_matched = False
        for slide_id, label in slide_label_map.items():
            # Match by prefix (embedding file may not have UUID)
            slide_prefix = slide_id.split(".")[0]  # Remove UUID suffix
            if name == slide_prefix or name.startswith(slide_prefix):
                embeddings = np.load(emb_file).astype(np.float32)
                embeddings_list.append(embeddings)
                labels.append(int(label))
                patient_ids.append(slide_prefix)
                logger.info(f"  {slide_prefix}: {embeddings.shape[0]} patches, label={label}")
                slide_matched = True
                break
        
        if slide_matched:
            continue
        
        # Fall back to patient_id matching
        parts = name.split("-")
        if len(parts) >= 3 and parts[0] == "TCGA":
            patient_id = "-".join(parts[:3])
        else:
            # Demo slide format - skip
            continue
        
        if patient_id in patient_label_map:
            embeddings = np.load(emb_file).astype(np.float32)
            embeddings_list.append(embeddings)
            labels.append(int(patient_label_map[patient_id]))
            patient_ids.append(patient_id)
            logger.info(f"  {patient_id}: {embeddings.shape[0]} patches, label={patient_label_map[patient_id]}")
        else:
            logger.debug(f"No label for {patient_id}")
    
    logger.info(f"Loaded {len(embeddings_list)} slides with labels")
    return embeddings_list, labels, patient_ids


class FocalLoss:
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        import torch
        import torch.nn.functional as F
        
        # pred is logit, apply sigmoid
        p = torch.sigmoid(pred)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal weight
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        
        loss = focal_weight * bce
        return loss.mean()


def oversample_minority(
    embeddings_list: List[np.ndarray],
    labels: List[int],
    patient_ids: List[str],
    random_seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Oversample minority class to balance the dataset.
    Uses random oversampling (replication).
    """
    np.random.seed(random_seed)
    
    # Find minority class
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    
    if len(pos_idx) == len(neg_idx):
        logger.info("Classes already balanced, no oversampling needed")
        return embeddings_list, labels, patient_ids
    
    if len(pos_idx) < len(neg_idx):
        minority_idx = pos_idx
        majority_count = len(neg_idx)
        minority_label = 1
    else:
        minority_idx = neg_idx
        majority_count = len(pos_idx)
        minority_label = 0
    
    # Calculate how many samples to add
    samples_to_add = majority_count - len(minority_idx)
    
    # Randomly sample from minority with replacement
    oversampled_idx = np.random.choice(minority_idx, size=samples_to_add, replace=True)
    
    # Create new lists
    new_embeddings = embeddings_list.copy()
    new_labels = labels.copy()
    new_patient_ids = patient_ids.copy()
    
    for idx in oversampled_idx:
        new_embeddings.append(embeddings_list[idx])
        new_labels.append(labels[idx])
        new_patient_ids.append(f"{patient_ids[idx]}_oversample")
    
    logger.info(f"Oversampled minority class (label={minority_label}): {len(minority_idx)} -> {majority_count}")
    logger.info(f"Total samples after oversampling: {len(new_labels)}")
    
    return new_embeddings, new_labels, new_patient_ids


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal decision threshold using Youden's J statistic.
    J = sensitivity + specificity - 1
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[best_idx]
    
    # Compute metrics at optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "youden_j": float(j_scores[best_idx]),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / len(y_true))
    }
    
    return optimal_threshold, metrics


def build_model(config: TrainingConfig):
    """Build CLAM model."""
    import torch
    import torch.nn as nn
    
    class GatedAttention(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_heads=1, dropout=0.25):
            super().__init__()
            self.n_heads = n_heads
            self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
            self.attention_weights = nn.Linear(hidden_dim, n_heads)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            V = self.attention_V(x)
            U = self.attention_U(x)
            A = self.attention_weights(self.dropout(V * U))
            A = torch.softmax(A, dim=0)
            return A
    
    class CLAM(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_heads, dropout):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.attention = GatedAttention(hidden_dim, hidden_dim // 2, n_heads, dropout)
            self.bag_classifier = nn.Sequential(
                nn.Linear(hidden_dim * n_heads, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, x, return_attention=True):
            h = self.feature_extractor(x)
            A = self.attention(h)
            M = torch.mm(A.T, h).view(-1)
            logit = self.bag_classifier(M)
            
            if return_attention:
                return logit, A.mean(dim=1)
            return logit, None
    
    return CLAM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.attention_heads,
        dropout=config.dropout
    )


def setup_device():
    """Setup computation device."""
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device


def train_and_evaluate(
    config: TrainingConfig,
    embeddings_list: List[np.ndarray],
    labels: List[int],
    patient_ids: List[str]
) -> Dict:
    """
    Train with leave-one-out CV and evaluate with all metrics.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score, confusion_matrix
    
    device = setup_device()
    
    # Optionally oversample before CV
    if config.use_oversampling:
        embeddings_list, labels, patient_ids = oversample_minority(
            embeddings_list, labels, patient_ids, config.random_seed
        )
    
    n_samples = len(embeddings_list)
    all_predictions = []
    all_true_labels = []
    all_patient_ids = []
    
    logger.info(f"\n{'='*60}")
    logger.info("LEAVE-ONE-OUT CROSS-VALIDATION")
    logger.info(f"{'='*60}")
    
    for i in range(n_samples):
        # Skip oversampled duplicates in test set
        if "_oversample" in patient_ids[i]:
            continue
            
        logger.info(f"\nFold {i+1}: Testing on {patient_ids[i]}")
        
        # Split data
        train_emb = [embeddings_list[j] for j in range(n_samples) if j != i]
        train_lab = [labels[j] for j in range(n_samples) if j != i]
        test_emb = embeddings_list[i]
        test_lab = labels[i]
        
        # Build model
        model = build_model(config)
        model.to(device)
        
        # Setup loss
        if config.use_focal_loss:
            criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        else:
            # Use class weights for BCE
            pos_count = sum(train_lab)
            neg_count = len(train_lab) - pos_count
            if pos_count > 0 and neg_count > 0:
                pos_weight = torch.tensor([neg_count / pos_count]).to(device)
            else:
                pos_weight = None
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(config.epochs):
            model.train()
            train_losses = []
            
            indices = np.random.permutation(len(train_emb))
            for idx in indices:
                x = torch.from_numpy(train_emb[idx]).float().to(device)
                y = torch.tensor([train_lab[idx]], dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                logit, _ = model(x)
                loss = criterion(logit.view(-1), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_loss = np.mean(train_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break
        
        # Restore best and predict
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(test_emb).float().to(device)
            logit, _ = model(x)
            prob = torch.sigmoid(logit).item()
        
        all_predictions.append(prob)
        all_true_labels.append(test_lab)
        all_patient_ids.append(patient_ids[i])
        
        pred_class = 1 if prob > 0.5 else 0
        status = "CORRECT" if pred_class == test_lab else "WRONG"
        logger.info(f"  Prediction: {prob:.4f}, True: {test_lab}, {status}")
    
    # Compute metrics
    y_true = np.array(all_true_labels)
    y_prob = np.array(all_predictions)
    y_pred = (y_prob >= 0.5).astype(int)
    
    results = {
        "config": {
            "label_type": config.label_type,
            "use_oversampling": config.use_oversampling,
            "use_focal_loss": config.use_focal_loss,
            "survival_threshold": config.survival_threshold
        },
        "n_samples": len(y_true),
        "class_distribution": {
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum())
        }
    }
    
    # Standard metrics at 0.5 threshold
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    results["metrics_threshold_0.5"] = {
        "auc_roc": float(auc),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
    
    # Optimal threshold metrics
    if config.use_optimal_threshold and len(np.unique(y_true)) > 1:
        opt_threshold, opt_metrics = find_optimal_threshold(y_true, y_prob)
        results["metrics_optimal_threshold"] = opt_metrics
    
    # Per-sample results
    results["predictions"] = [
        {"patient_id": pid, "true_label": int(t), "predicted_prob": float(p)}
        for pid, t, p in zip(all_patient_ids, y_true, y_prob)
    ]
    
    return results


def print_results(results: Dict):
    """Print results summary."""
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Label type: {results['config']['label_type']}")
    logger.info(f"  Oversampling: {results['config']['use_oversampling']}")
    logger.info(f"  Focal loss: {results['config']['use_focal_loss']}")
    
    logger.info(f"\nData:")
    logger.info(f"  Total samples: {results['n_samples']}")
    logger.info(f"  Positive: {results['class_distribution']['positive']}")
    logger.info(f"  Negative: {results['class_distribution']['negative']}")
    
    logger.info(f"\nMetrics at threshold=0.5:")
    m = results["metrics_threshold_0.5"]
    logger.info(f"  AUC-ROC: {m['auc_roc']:.4f}")
    logger.info(f"  Accuracy: {m['accuracy']:.4f}")
    logger.info(f"  Sensitivity: {m['sensitivity']:.4f}")
    logger.info(f"  Specificity: {m['specificity']:.4f}")
    logger.info(f"  PPV: {m['ppv']:.4f}")
    logger.info(f"  NPV: {m['npv']:.4f}")
    logger.info(f"  Confusion: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")
    
    if "metrics_optimal_threshold" in results:
        logger.info(f"\nMetrics at optimal threshold:")
        m = results["metrics_optimal_threshold"]
        logger.info(f"  Optimal threshold: {m['optimal_threshold']:.4f}")
        logger.info(f"  Youden's J: {m['youden_j']:.4f}")
        logger.info(f"  Sensitivity: {m['sensitivity']:.4f}")
        logger.info(f"  Specificity: {m['specificity']:.4f}")
        logger.info(f"  Accuracy: {m['accuracy']:.4f}")


def load_direct_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load direct labels from labels.csv (slide-level format).
    Format: slide_id,patient_id,label,platinum_status
    """
    df = pd.read_csv(labels_path)
    logger.info(f"Loaded direct labels for {len(df)} slides")
    
    # Count distribution
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    ratio = max(pos, neg) / min(pos, neg) if min(pos, neg) > 0 else float("inf")
    
    logger.info(f"Distribution: positive={pos}, negative={neg}, ratio={ratio:.2f}:1")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train CLAM with alternative approaches")
    parser.add_argument("--label_type", choices=["platinum", "survival", "recurrence", "direct"],
                       default="direct", help="Label type to use (direct=use labels.csv directly)")
    parser.add_argument("--survival_threshold", type=int, default=36,
                       help="OS threshold in months for survival label")
    parser.add_argument("--oversample", action="store_true",
                       help="Use random oversampling for minority class")
    parser.add_argument("--focal_loss", action="store_true",
                       help="Use focal loss instead of BCE")
    parser.add_argument("--optimal_threshold", action="store_true",
                       help="Report metrics at optimal threshold (Youden's J)")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory with embeddings and labels")
    parser.add_argument("--clinical_file", type=str, default=None,
                       help="Path to clinical CSV file")
    parser.add_argument("--labels_file", type=str, default=None,
                       help="Path to labels.csv file (direct slide-level labels)")
    parser.add_argument("--output_dir", type=str, default="outputs/alternative_training",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_dir / "data"
    
    embeddings_dir = data_dir / "embeddings"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = TrainingConfig(
        label_type=args.label_type,
        survival_threshold=args.survival_threshold,
        use_oversampling=args.oversample,
        use_focal_loss=args.focal_loss,
        use_optimal_threshold=args.optimal_threshold
    )
    
    logger.info(f"{'='*60}")
    logger.info("ALTERNATIVE TRAINING APPROACHES FOR TCGA OVARIAN CANCER")
    logger.info(f"{'='*60}")
    logger.info(f"Embeddings: {embeddings_dir}")
    logger.info(f"Label type: {config.label_type}")
    
    # Load data based on label type
    if args.label_type == "direct" or args.labels_file:
        # Use direct labels.csv format
        labels_path = Path(args.labels_file) if args.labels_file else (data_dir / "tcga_full" / "labels.csv")
        if not labels_path.exists():
            labels_path = data_dir / "labels.csv"
        
        logger.info(f"Labels file: {labels_path}")
        labels_df = load_direct_labels(labels_path)
        clinical_df = labels_df  # Use same df for compatibility
        
    else:
        # Use clinical data with derived labels
        if args.clinical_file:
            clinical_path = Path(args.clinical_file)
        else:
            # Try TCGA full clinical data first
            tcga_clinical = Path.home() / "med-gemma-hackathon/data/tcga_full/clinical.csv"
            if tcga_clinical.exists():
                clinical_path = tcga_clinical
            else:
                clinical_path = data_dir / "clinical.csv"
                if not clinical_path.exists():
                    clinical_path = data_dir / "labels.csv"
        
        logger.info(f"Clinical data: {clinical_path}")
        clinical_df = load_clinical_data(clinical_path)
        labels_df = create_labels(clinical_df, config.label_type, config.survival_threshold)
    
    embeddings_list, labels, patient_ids = load_embeddings_with_labels(
        embeddings_dir, labels_df, clinical_df
    )
    
    if len(embeddings_list) == 0:
        logger.error("No embeddings found with matching labels!")
        logger.error("Available embeddings need TCGA slide IDs matching labels data.")
        return
    
    if len(embeddings_list) < 3:
        logger.warning(f"Only {len(embeddings_list)} samples available - results will be limited")
    
    # Check class balance
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if pos_count == 0 or neg_count == 0:
        logger.error(f"Only one class present in data (pos={pos_count}, neg={neg_count})!")
        logger.error("Cannot train a meaningful classifier. Need at least 1 sample per class.")
        return
    
    # Train and evaluate
    results = train_and_evaluate(config, embeddings_list, labels, patient_ids)
    
    # Print and save results
    print_results(results)
    
    output_file = output_dir / f"results_{config.label_type}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
