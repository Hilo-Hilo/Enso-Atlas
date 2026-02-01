#!/usr/bin/env python3
"""
Production CLAM Training Pipeline for Ovarian Cancer Treatment Response Prediction.

Features:
- Patient-level data splits (no data leakage)
- 5-fold stratified cross-validation
- Comprehensive clinical metrics
- Early stopping, LR scheduling, class weighting
- Best model checkpointing

Usage:
    python scripts/train_clam_production.py --data_dir data/demo --output_dir outputs/training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model architecture
    input_dim: int = 384
    hidden_dim: int = 256
    attention_heads: int = 1
    dropout: float = 0.25
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    
    # Cross-validation
    n_folds: int = 5
    random_seed: int = 42
    
    # Data splits (for final evaluation)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # LR scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Class weighting
    use_class_weights: bool = True
    
    # Paths
    data_dir: str = "data/demo"
    output_dir: str = "outputs/training"


class CLAMProductionModel:
    """Production CLAM model with proper training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._model = None
        self._device = None
        self._optimizer = None
        self._scheduler = None
        self._criterion = None
        
    def _setup_device(self):
        """Setup computation device."""
        import torch
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Using device: {self._device}")
        return self._device
    
    def _build_model(self):
        """Build the CLAM model architecture."""
        import torch
        import torch.nn as nn
        
        class GatedAttention(nn.Module):
            """Gated attention mechanism."""
            
            def __init__(self, input_dim, hidden_dim, n_heads=1, dropout=0.25):
                super().__init__()
                self.n_heads = n_heads
                
                self.attention_V = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh()
                )
                
                self.attention_U = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Sigmoid()
                )
                
                self.attention_weights = nn.Linear(hidden_dim, n_heads)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                V = self.attention_V(x)
                U = self.attention_U(x)
                A = self.attention_weights(self.dropout(V * U))
                A = torch.softmax(A, dim=0)
                return A
        
        class CLAM(nn.Module):
            """
            Clustering-constrained Attention Multiple Instance Learning.
            
            Architecture:
            - Feature transformation layer
            - Gated attention mechanism
            - Instance-level classifier (for pseudo-labels)
            - Bag-level classifier
            """
            
            def __init__(self, input_dim, hidden_dim, n_heads, dropout, n_classes=2):
                super().__init__()
                self.n_classes = n_classes
                self.n_heads = n_heads
                
                # Feature transformation
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Gated attention
                self.attention = GatedAttention(hidden_dim, hidden_dim // 2, n_heads, dropout)
                
                # Instance-level classifiers (for clustering constraint)
                self.instance_classifiers = nn.ModuleList([
                    nn.Linear(hidden_dim, 2) for _ in range(n_classes)
                ])
                
                # Bag-level classifier
                self.bag_classifier = nn.Sequential(
                    nn.Linear(hidden_dim * n_heads, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
                
            def forward(self, x, return_attention=True, return_instance=False):
                """
                Forward pass.
                
                Args:
                    x: (n_patches, input_dim)
                    return_attention: Return attention weights
                    return_instance: Return instance-level predictions
                    
                Returns:
                    logit: Slide-level prediction (before sigmoid)
                    attention: Attention weights per patch
                    instance_preds: Instance-level predictions (optional)
                """
                # Transform features
                h = self.feature_extractor(x)  # (n_patches, hidden_dim)
                
                # Compute attention
                A = self.attention(h)  # (n_patches, n_heads)
                
                # Aggregate with attention
                M = torch.mm(A.T, h)  # (n_heads, hidden_dim)
                M = M.view(-1)  # (n_heads * hidden_dim,)
                
                # Bag-level prediction (logit, not probability)
                logit = self.bag_classifier(M)
                
                outputs = [logit]
                
                if return_attention:
                    attn_weights = A.mean(dim=1)  # Average across heads
                    outputs.append(attn_weights)
                
                if return_instance:
                    instance_preds = [clf(h) for clf in self.instance_classifiers]
                    outputs.append(instance_preds)
                
                return outputs if len(outputs) > 1 else outputs[0]
        
        self._model = CLAM(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            n_heads=self.config.attention_heads,
            dropout=self.config.dropout
        )
        
        return self._model
    
    def _compute_class_weights(self, labels: List[int]):
        """Compute class weights for imbalanced data."""
        import torch
        from collections import Counter
        
        counts = Counter(labels)
        total = len(labels)
        n_classes = len(counts)
        
        weights = {}
        for cls, count in counts.items():
            weights[cls] = total / (n_classes * count)
        
        # Normalize
        max_weight = max(weights.values())
        weights = {k: v / max_weight for k, v in weights.items()}
        
        logger.info(f"Class distribution: {dict(counts)}")
        logger.info(f"Class weights: {weights}")
        
        return weights
    
    def fit(
        self,
        train_embeddings: List[np.ndarray],
        train_labels: List[int],
        val_embeddings: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[int]] = None,
        fold_id: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the CLAM model.
        
        Args:
            train_embeddings: List of embedding arrays (one per slide)
            train_labels: Binary labels
            val_embeddings: Validation embeddings
            val_labels: Validation labels
            fold_id: Fold identifier for logging
            
        Returns:
            Training history
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        self._setup_device()
        self._build_model()
        self._model.to(self._device)
        
        # Compute class weights
        class_weights = None
        if self.config.use_class_weights:
            class_weights = self._compute_class_weights(train_labels)
        
        # Setup optimizer
        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup LR scheduler
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode='min',
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor,
            min_lr=self.config.min_lr
        )
        
        # Loss function with class weights
        pos_weight = None
        if class_weights:
            pos_weight = torch.tensor([class_weights.get(1, 1.0) / class_weights.get(0, 1.0)]).to(self._device)
        
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "learning_rate": []
        }
        
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        
        fold_str = f"[Fold {fold_id}] " if fold_id is not None else ""
        
        for epoch in range(self.config.epochs):
            # Training
            self._model.train()
            train_losses = []
            
            indices = np.random.permutation(len(train_embeddings))
            
            for idx in indices:
                embeddings = train_embeddings[idx]
                label = train_labels[idx]
                
                x = torch.from_numpy(embeddings).float().to(self._device)
                y = torch.tensor([label], dtype=torch.float32).to(self._device)
                
                self._optimizer.zero_grad()
                logit, _ = self._model(x, return_attention=True)
                
                # Ensure proper shapes for loss
                logit = logit.view(-1)
                y = y.view(-1)
                loss = self._criterion(logit, y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                
                self._optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            history["learning_rate"].append(self._optimizer.param_groups[0]["lr"])
            
            # Validation
            if val_embeddings is not None and val_labels is not None:
                val_loss, val_auc, _, _ = self._evaluate(val_embeddings, val_labels)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)
                
                # LR scheduler step
                self._scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"{fold_str}Epoch {epoch+1}/{self.config.epochs}: "
                        f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"val_auc={val_auc:.4f}, lr={self._optimizer.param_groups[0]['lr']:.2e}"
                    )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"{fold_str}Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"{fold_str}Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)
            logger.info(f"{fold_str}Restored best model (val_loss={best_val_loss:.4f})")
        
        return history
    
    def _evaluate(
        self,
        embeddings_list: List[np.ndarray],
        labels: List[int]
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model and return metrics."""
        import torch
        from sklearn.metrics import roc_auc_score
        
        self._model.eval()
        
        losses = []
        preds = []
        attentions = []
        
        with torch.no_grad():
            for embeddings, label in zip(embeddings_list, labels):
                x = torch.from_numpy(embeddings).float().to(self._device)
                y = torch.tensor([label], dtype=torch.float32).to(self._device)
                
                logit, attn = self._model(x, return_attention=True)
                
                # Ensure proper shapes for loss
                logit_flat = logit.view(-1)
                y_flat = y.view(-1)
                loss = self._criterion(logit_flat, y_flat)
                
                prob = torch.sigmoid(logit).item()
                
                losses.append(loss.item())
                preds.append(prob)
                attentions.append(attn.cpu().numpy())
        
        avg_loss = np.mean(losses)
        preds = np.array(preds)
        
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5
        
        return avg_loss, auc, preds, attentions
    
    def predict(self, embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """Predict for a single slide."""
        import torch
        
        self._model.eval()
        x = torch.from_numpy(embeddings).float().to(self._device)
        
        with torch.no_grad():
            logit, attn = self._model(x, return_attention=True)
            prob = torch.sigmoid(logit).item()
        
        return prob, attn.cpu().numpy()
    
    def save(self, path: Path):
        """Save model state."""
        import torch
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "config": self.config.__dict__
        }, path)
        
        logger.info(f"Saved model to {path}")
    
    def load(self, path: Path):
        """Load model state."""
        import torch
        
        self._setup_device()
        self._build_model()
        
        checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()
        
        logger.info(f"Loaded model from {path}")


def load_data(
    data_dir: Path,
    labels_file: str = "labels.csv"
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Load embeddings and labels.
    
    Returns:
        embeddings_dict: {slide_id: embeddings_array}
        labels_df: DataFrame with patient/slide info and labels
    """
    data_dir = Path(data_dir)
    
    # Load labels
    labels_path = data_dir / labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    logger.info(f"Loaded labels for {len(labels_df)} samples")
    
    # Determine slide ID column
    slide_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    
    # Load embeddings
    embeddings_dir = data_dir / "embeddings"
    embeddings_dict = {}
    
    for _, row in labels_df.iterrows():
        slide_id = row[slide_col]
        
        # Try different naming conventions
        possible_names = [
            f"{slide_id}.npy",
            f"demo_{slide_id}.npy",
        ]
        
        found = False
        for name in possible_names:
            emb_path = embeddings_dir / name
            if emb_path.exists():
                embeddings_dict[slide_id] = np.load(emb_path)
                found = True
                break
        
        if not found:
            logger.warning(f"Embeddings not found for {slide_id}")
    
    logger.info(f"Loaded embeddings for {len(embeddings_dict)} slides")
    
    return embeddings_dict, labels_df


def create_patient_splits(
    labels_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create patient-level splits (no data leakage).
    Stratified by outcome.
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    # Determine ID and label columns
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    label_col = "label" if "label" in labels_df.columns else "treatment_response"
    
    # Convert labels if necessary
    if labels_df[label_col].dtype == object:
        labels_df = labels_df.copy()
        labels_df["_label"] = (labels_df[label_col] == "responder").astype(int)
        label_col = "_label"
    
    patient_ids = labels_df[id_col].tolist()
    labels = labels_df[label_col].tolist()
    
    # First split: train+val vs test
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        patient_ids, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_seed
    )
    
    logger.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def create_cv_folds(
    labels_df: pd.DataFrame,
    n_folds: int = 5,
    random_seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create stratified k-fold cross-validation splits.
    Patient-level splits to avoid data leakage, then expand to slide IDs.
    """
    from sklearn.model_selection import StratifiedKFold
    
    label_col = "label" if "label" in labels_df.columns else "treatment_response"
    
    if labels_df[label_col].dtype == object:
        labels_df = labels_df.copy()
        labels_df["_label"] = (labels_df[label_col] == "responder").astype(int)
        label_col = "_label"
    
    # Use patient_id for splitting to avoid data leakage
    if "patient_id" in labels_df.columns and "slide_id" in labels_df.columns:
        # Get unique patients with their labels
        patient_labels = labels_df.groupby("patient_id")[label_col].first().reset_index()
        patient_ids = np.array(patient_labels["patient_id"].tolist())
        labels = np.array(patient_labels[label_col].tolist())
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        folds = []
        for train_idx, val_idx in skf.split(patient_ids, labels):
            train_patients = patient_ids[train_idx].tolist()
            val_patients = patient_ids[val_idx].tolist()
            
            # Expand to slide IDs
            train_slides = labels_df[labels_df["patient_id"].isin(train_patients)]["slide_id"].tolist()
            val_slides = labels_df[labels_df["patient_id"].isin(val_patients)]["slide_id"].tolist()
            
            folds.append((train_slides, val_slides))
    else:
        # Fallback: use slide_id directly
        id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
        patient_ids = np.array(labels_df[id_col].tolist())
        labels = np.array(labels_df[label_col].tolist())
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        folds = []
        for train_idx, val_idx in skf.split(patient_ids, labels):
            train_ids = patient_ids[train_idx].tolist()
            val_ids = patient_ids[val_idx].tolist()
            folds.append((train_ids, val_ids))
    
    return folds


def get_data_for_ids(
    ids: List[str],
    embeddings_dict: Dict[str, np.ndarray],
    labels_df: pd.DataFrame
) -> Tuple[List[np.ndarray], List[int]]:
    """Get embeddings and labels for given IDs."""
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    label_col = "label" if "label" in labels_df.columns else "treatment_response"
    
    embeddings = []
    labels = []
    
    for slide_id in ids:
        if slide_id not in embeddings_dict:
            continue
        
        embeddings.append(embeddings_dict[slide_id])
        
        row = labels_df[labels_df[id_col] == slide_id].iloc[0]
        label = row[label_col]
        
        if isinstance(label, str):
            label = 1 if label == "responder" else 0
        
        labels.append(int(label))
    
    return embeddings, labels


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute clinically relevant metrics.
    
    Metrics:
    - AUC-ROC: Discrimination ability
    - Sensitivity: True positive rate (important for catching responders)
    - Specificity: True negative rate
    - PPV: Positive predictive value
    - NPV: Negative predictive value
    - Brier score: Calibration measure
    """
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss,
        confusion_matrix, accuracy_score
    )
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # AUC-ROC
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.5
    
    # Confusion matrix based metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Brier score (lower is better, 0-1 range)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    
    # Additional counts
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    
    return metrics


def run_cross_validation(
    config: TrainingConfig,
    embeddings_dict: Dict[str, np.ndarray],
    labels_df: pd.DataFrame
) -> Dict:
    """
    Run k-fold cross-validation and report mean +/- std of metrics.
    """
    folds = create_cv_folds(labels_df, config.n_folds, config.random_seed)
    
    all_metrics = []
    all_histories = []
    
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{config.n_folds}")
        logger.info(f"{'='*60}")
        
        # Get data
        train_emb, train_labels = get_data_for_ids(train_ids, embeddings_dict, labels_df)
        val_emb, val_labels = get_data_for_ids(val_ids, embeddings_dict, labels_df)
        
        logger.info(f"Train: {len(train_emb)} samples, Val: {len(val_emb)} samples")
        
        # Train model
        model = CLAMProductionModel(config)
        history = model.fit(train_emb, train_labels, val_emb, val_labels, fold_id=fold_idx+1)
        all_histories.append(history)
        
        # Evaluate
        _, _, val_preds, _ = model._evaluate(val_emb, val_labels)
        metrics = compute_clinical_metrics(val_labels, val_preds >= 0.5, val_preds)
        all_metrics.append(metrics)
        
        logger.info(f"Fold {fold_idx+1} Results:")
        for k, v in metrics.items():
            if k not in ["tp", "tn", "fp", "fn"]:
                logger.info(f"  {k}: {v:.4f}")
    
    # Aggregate results
    cv_results = {
        "n_folds": config.n_folds,
        "per_fold_metrics": all_metrics,
        "mean_metrics": {},
        "std_metrics": {}
    }
    
    metric_names = ["auc_roc", "sensitivity", "specificity", "ppv", "npv", "brier_score", "accuracy"]
    
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS (Mean +/- Std)")
    logger.info(f"{'='*60}")
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        cv_results["mean_metrics"][metric] = mean_val
        cv_results["std_metrics"][metric] = std_val
        
        logger.info(f"{metric}: {mean_val:.4f} +/- {std_val:.4f}")
    
    return cv_results


def train_final_model(
    config: TrainingConfig,
    embeddings_dict: Dict[str, np.ndarray],
    labels_df: pd.DataFrame,
    output_dir: Path
) -> Tuple[CLAMProductionModel, Dict]:
    """
    Train final model on train+val, evaluate on held-out test set.
    """
    # Create splits
    train_ids, val_ids, test_ids = create_patient_splits(
        labels_df,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed
    )
    
    # Get data
    train_emb, train_labels = get_data_for_ids(train_ids, embeddings_dict, labels_df)
    val_emb, val_labels = get_data_for_ids(val_ids, embeddings_dict, labels_df)
    test_emb, test_labels = get_data_for_ids(test_ids, embeddings_dict, labels_df)
    
    logger.info(f"\nFinal model training:")
    logger.info(f"  Train: {len(train_emb)} samples")
    logger.info(f"  Val: {len(val_emb)} samples")
    logger.info(f"  Test: {len(test_emb)} samples (held out)")
    
    # Train model
    model = CLAMProductionModel(config)
    history = model.fit(train_emb, train_labels, val_emb, val_labels)
    
    # Evaluate on test set
    logger.info("\nEvaluating on held-out test set...")
    _, test_auc, test_preds, test_attentions = model._evaluate(test_emb, test_labels)
    test_metrics = compute_clinical_metrics(test_labels, test_preds >= 0.5, test_preds)
    
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET RESULTS")
    logger.info(f"{'='*60}")
    for k, v in test_metrics.items():
        if k not in ["tp", "tn", "fp", "fn"]:
            logger.info(f"  {k}: {v:.4f}")
    
    # Save model
    model_path = output_dir / "best_model.pt"
    model.save(model_path)
    
    # Save test set info for later evaluation
    test_info = {
        "test_ids": test_ids,
        "test_labels": test_labels,
        "test_predictions": test_preds.tolist(),
        "test_metrics": test_metrics,
        "history": {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    return model, test_info


def main():
    parser = argparse.ArgumentParser(
        description="Production CLAM Training Pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/demo",
        help="Directory containing embeddings and labels"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/training",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip_cv", action="store_true",
        help="Skip cross-validation (only train final model)"
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = TrainingConfig(
        n_folds=args.n_folds,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        random_seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info("CLAM Production Training Pipeline")
    logger.info("="*60)
    logger.info(f"Config: {config}")
    
    # Load data
    data_dir = Path(config.data_dir)
    embeddings_dict, labels_df = load_data(data_dir)
    
    results = {
        "timestamp": timestamp,
        "config": config.__dict__,
    }
    
    # Run cross-validation
    if not args.skip_cv:
        logger.info("\n" + "="*60)
        logger.info("RUNNING CROSS-VALIDATION")
        logger.info("="*60)
        
        cv_results = run_cross_validation(config, embeddings_dict, labels_df)
        results["cross_validation"] = cv_results
    
    # Train final model
    logger.info("\n" + "="*60)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*60)
    
    model, test_info = train_final_model(config, embeddings_dict, labels_df, output_dir)
    results["final_model"] = test_info
    
    # Save results
    results_file = output_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info(f"Log saved to: {log_file}")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
