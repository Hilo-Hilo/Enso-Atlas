#!/usr/bin/env python3
"""
Training script for Bevacizumab Treatment Response Prediction.

Dataset: Ovarian cancer bevacizumab response cohort
- 286 slides from 126 patients
- Label: 1=effective (56%), 0=invalid (44%)
- Pre-defined train/val/test splits

Usage:
    python scripts/train_bevacizumab.py --data_dir data/ovarian_bev --output_dir outputs/bevacizumab
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

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
    """Training configuration for bevacizumab response prediction."""
    # Model architecture
    input_dim: int = 384  # MedGemma embedding dimension
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
    
    # LR scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Class weighting (less needed - balanced dataset!)
    use_class_weights: bool = True
    
    # Paths
    data_dir: str = "data/ovarian_bev"
    output_dir: str = "outputs/bevacizumab"


def load_bevacizumab_data(
    data_dir: Path
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Load bevacizumab embeddings and clinical labels.
    
    Uses pre-defined train/val/test splits from clinical.csv.
    """
    clinical_path = data_dir / "clinical.csv"
    embeddings_dir = data_dir / "embeddings"
    
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical file not found: {clinical_path}")
    
    df = pd.read_csv(clinical_path)
    logger.info(f"Loaded clinical data for {len(df)} slides")
    
    # Check label distribution
    label_counts = df["label"].value_counts()
    logger.info(f"Label distribution: {dict(label_counts)}")
    logger.info(f"Split distribution: {dict(df[split].value_counts())}")
    
    # Load embeddings
    embeddings_dict = {}
    missing = 0
    
    for _, row in df.iterrows():
        slide_id = row["slide_id"]
        emb_path = embeddings_dir / f"{slide_id}.npy"
        
        if emb_path.exists():
            embeddings_dict[slide_id] = np.load(emb_path)
        else:
            missing += 1
    
    logger.info(f"Loaded {len(embeddings_dict)} embeddings, {missing} missing")
    
    return embeddings_dict, df


def prepare_split_data(
    embeddings_dict: Dict[str, np.ndarray],
    df: pd.DataFrame,
    split: str
) -> Tuple[List[np.ndarray], List[int]]:
    """Get embeddings and labels for a specific split."""
    split_df = df[df["split"] == split]
    
    embeddings = []
    labels = []
    
    for _, row in split_df.iterrows():
        slide_id = row["slide_id"]
        if slide_id in embeddings_dict:
            embeddings.append(embeddings_dict[slide_id])
            labels.append(int(row["label"]))
    
    return embeddings, labels


class CLAMAttention(object):
    """CLAM-style attention network for MIL."""
    
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
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU")
    
    def _build_model(self):
        """Build attention-based MIL model."""
        import torch
        import torch.nn as nn
        
        class AttentionMIL(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout=0.25):
                super().__init__()
                
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # x: (n_instances, input_dim)
                h = self.encoder(x)  # (n_instances, hidden_dim)
                
                # Attention weights
                a = self.attention(h)  # (n_instances, 1)
                a = torch.softmax(a, dim=0)
                
                # Weighted aggregation
                z = torch.sum(a * h, dim=0, keepdim=True)  # (1, hidden_dim)
                
                # Classification
                y = self.classifier(z)  # (1, 1)
                
                return y, a
        
        self._model = AttentionMIL(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.dropout
        ).to(self._device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters()):,}")
    
    def _compute_class_weights(self, labels: List[int]):
        """Compute class weights for imbalanced data."""
        import torch
        counts = Counter(labels)
        total = len(labels)
        weights = {cls: total / (2 * count) for cls, count in counts.items()}
        logger.info(f"Class weights: {weights}")
        return torch.tensor(weights[1], dtype=torch.float32).to(self._device)
    
    def train(
        self,
        train_embeddings: List[np.ndarray],
        train_labels: List[int],
        val_embeddings: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict:
        """Train the model."""
        import torch
        import torch.nn as nn
        
        self._setup_device()
        self._build_model()
        
        # Setup optimizer and scheduler
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor,
            min_lr=self.config.min_lr
        )
        
        # Class weights
        if self.config.use_class_weights:
            pos_weight = self._compute_class_weights(train_labels)
            self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self._criterion = nn.BCELoss()
        
        # Training loop
        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        
        for epoch in range(self.config.epochs):
            # Training
            self._model.train()
            train_loss = 0.0
            
            indices = np.random.permutation(len(train_embeddings))
            
            for idx in indices:
                self._optimizer.zero_grad()
                
                emb = torch.tensor(train_embeddings[idx], dtype=torch.float32).to(self._device)
                label = torch.tensor([[train_labels[idx]]], dtype=torch.float32).to(self._device)
                
                output, _ = self._model(emb)
                loss = self._criterion(output, label)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_embeddings)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_embeddings is not None and val_labels is not None:
                val_loss, val_auc, _, _ = self._evaluate(val_embeddings, val_labels)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)
                
                self._scheduler.step(val_loss)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
                    )
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if val_embeddings is not None:
            self._model.load_state_dict(best_state)
            logger.info(f"Restored best model from epoch {best_epoch+1} (AUC={best_val_auc:.4f})")
        
        return {
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "history": history
        }
    
    def _evaluate(
        self,
        embeddings_list: List[np.ndarray],
        labels: List[int]
    ) -> Tuple[float, float, List[float], List[int]]:
        """Evaluate the model."""
        import torch
        from sklearn.metrics import roc_auc_score
        
        self._model.eval()
        
        total_loss = 0.0
        preds = []
        
        with torch.no_grad():
            for emb, label in zip(embeddings_list, labels):
                emb_t = torch.tensor(emb, dtype=torch.float32).to(self._device)
                label_t = torch.tensor([[label]], dtype=torch.float32).to(self._device)
                
                output, _ = self._model(emb_t)
                loss = self._criterion(output, label_t)
                
                total_loss += loss.item()
                preds.append(output.item())
        
        avg_loss = total_loss / len(embeddings_list)
        
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5
        
        return avg_loss, auc, preds, labels
    
    def save(self, path: Path):
        """Save model checkpoint."""
        import torch
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "config": self.config
        }, path)
        logger.info(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train bevacizumab response predictor")
    parser.add_argument("--data_dir", type=str, default="data/ovarian_bev")
    parser.add_argument("--output_dir", type=str, default="outputs/bevacizumab")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading bevacizumab data...")
    embeddings_dict, df = load_bevacizumab_data(data_dir)
    
    if len(embeddings_dict) == 0:
        logger.error("No embeddings found! Run embedding generation first.")
        sys.exit(1)
    
    # Prepare splits
    train_emb, train_labels = prepare_split_data(embeddings_dict, df, "train")
    val_emb, val_labels = prepare_split_data(embeddings_dict, df, "val")
    test_emb, test_labels = prepare_split_data(embeddings_dict, df, "test")
    
    logger.info(f"Train: {len(train_emb)} slides, Val: {len(val_emb)}, Test: {len(test_emb)}")
    
    # Train model
    config = TrainingConfig(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr
    )
    
    model = CLAMAttention(config)
    result = model.train(train_emb, train_labels, val_emb, val_labels)
    
    # Test evaluation
    test_loss, test_auc, test_preds, _ = model._evaluate(test_emb, test_labels)
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    # Save model and results
    model.save(output_dir / "best_model.pt")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(config),
        "training": result,
        "test_auc": test_auc,
        "test_preds": test_preds,
        "test_labels": test_labels
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Final Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
