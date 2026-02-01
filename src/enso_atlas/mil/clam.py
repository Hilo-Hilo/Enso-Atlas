"""
CLAM - Clustering-constrained Attention Multiple Instance Learning.

Implementation based on:
"Data-efficient and weakly supervised computational pathology on whole-slide images"
Lu et al., Nature Biomedical Engineering, 2021

This module provides the attention-based MIL head for slide-level classification
with interpretable attention weights for evidence generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MILConfig:
    """MIL head configuration."""
    architecture: str = "clam"
    hidden_dim: int = 256
    attention_heads: int = 1
    dropout: float = 0.25
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20
    input_dim: int = 384  # Path Foundation embedding dim


class AttentionMIL:
    """
    Basic Attention-based Multiple Instance Learning.
    
    Simple baseline that learns attention weights over patch embeddings.
    """
    
    def __init__(self, config: MILConfig):
        self.config = config
        self._model = None
        self._device = None
    
    def _build_model(self):
        """Build the attention MIL model."""
        import torch
        import torch.nn as nn
        
        input_dim = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        
        class AttentionMILModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout):
                super().__init__()
                
                self.attention = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # x: (n_patches, input_dim)
                
                # Compute attention weights
                attn_scores = self.attention(x)  # (n_patches, 1)
                attn_weights = torch.softmax(attn_scores, dim=0)  # (n_patches, 1)
                
                # Weighted aggregation
                slide_embedding = torch.sum(attn_weights * x, dim=0)  # (input_dim,)
                
                # Classification
                logit = self.classifier(slide_embedding)
                
                return logit, attn_weights.squeeze()
        
        self._model = AttentionMILModel(
            input_dim, hidden_dim, self.config.dropout
        )
        
        return self._model


class CLAMClassifier:
    """
    CLAM (Clustering-constrained Attention MIL) classifier.
    
    Provides:
    - Slide-level classification (e.g., responder vs non-responder)
    - Per-patch attention weights for evidence
    - Instance-level pseudo-labels for refinement
    """
    
    def __init__(self, config: MILConfig):
        self.config = config
        self._model = None
        self._device = None
        self._is_trained = False
    
    def _build_model(self):
        """Build the CLAM model."""
        import torch
        import torch.nn as nn
        
        input_dim = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        n_heads = self.config.attention_heads
        
        class GatedAttention(nn.Module):
            """Gated attention mechanism for CLAM."""
            
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
                # x: (n_patches, input_dim)
                V = self.attention_V(x)  # (n_patches, hidden_dim)
                U = self.attention_U(x)  # (n_patches, hidden_dim)
                
                # Gated attention
                A = self.attention_weights(V * U)  # (n_patches, n_heads)
                A = torch.softmax(A, dim=0)  # Normalize over patches
                
                return A
        
        class CLAMModel(nn.Module):
            """Full CLAM model with gated attention and instance clustering."""
            
            def __init__(self, input_dim, hidden_dim, n_heads, dropout, n_classes=2):
                super().__init__()
                self.n_classes = n_classes
                
                # Feature transformation
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Gated attention
                self.attention = GatedAttention(hidden_dim, hidden_dim // 2, n_heads, dropout)
                
                # Instance-level classifier (for pseudo-labels)
                self.instance_classifier = nn.ModuleList([
                    nn.Linear(hidden_dim, 2) for _ in range(n_classes)
                ])
                
                # Bag-level classifier
                self.bag_classifier = nn.Sequential(
                    nn.Linear(hidden_dim * n_heads, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x, return_attention=True):
                # x: (n_patches, input_dim)
                
                # Transform features
                h = self.feature_extractor(x)  # (n_patches, hidden_dim)
                
                # Compute attention
                A = self.attention(h)  # (n_patches, n_heads)
                
                # Aggregate with attention
                M = torch.mm(A.T, h)  # (n_heads, hidden_dim)
                M = M.view(-1)  # (n_heads * hidden_dim,)
                
                # Bag-level prediction
                logit = self.bag_classifier(M)
                
                if return_attention:
                    # Average attention across heads
                    attn_weights = A.mean(dim=1)  # (n_patches,)
                    return logit, attn_weights
                
                return logit
        
        self._model = CLAMModel(
            input_dim, hidden_dim, n_heads, self.config.dropout
        )
        
        return self._model
    
    def _setup_device(self):
        """Setup computation device."""
        import torch
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        return self._device
    
    def load(self, path: str | Path) -> None:
        """Load a trained model from disk."""
        import torch
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self._setup_device()
        self._build_model()
        
        state_dict = torch.load(path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._is_trained = True
        
        logger.info(f"Loaded CLAM model from {path}")
    
    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        import torch
        
        if self._model is None:
            raise RuntimeError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self._model.state_dict(), path)
        logger.info(f"Saved CLAM model to {path}")
    
    def fit(
        self,
        embeddings_list: List[np.ndarray],
        labels: List[int],
        val_embeddings: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the CLAM model.
        
        Args:
            embeddings_list: List of embeddings arrays, one per slide
            labels: Binary labels (0 or 1) for each slide
            val_embeddings: Optional validation embeddings
            val_labels: Optional validation labels
            
        Returns:
            Training history dictionary
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        
        self._setup_device()
        self._build_model()
        self._model.to(self._device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.BCELoss()
        
        # Training history
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.epochs):
            self._model.train()
            train_losses = []
            
            # Shuffle training data
            indices = np.random.permutation(len(embeddings_list))
            
            for idx in tqdm(indices, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                embeddings = embeddings_list[idx]
                label = labels[idx]
                
                # Convert to tensor
                x = torch.from_numpy(embeddings).float().to(self._device)
                y = torch.tensor([label], dtype=torch.float32).to(self._device)
                
                # Forward pass
                optimizer.zero_grad()
                pred, _ = self._model(x)
                
                # Compute loss
                loss = criterion(pred.squeeze(), y.squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_embeddings is not None and val_labels is not None:
                val_loss, val_auc = self._validate(val_embeddings, val_labels, criterion)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)
                
                logger.info(
                    f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        self._is_trained = True
        return history
    
    def _validate(
        self,
        embeddings_list: List[np.ndarray],
        labels: List[int],
        criterion,
    ) -> Tuple[float, float]:
        """Run validation and compute metrics."""
        import torch
        from sklearn.metrics import roc_auc_score
        
        self._model.eval()
        
        losses = []
        preds = []
        
        with torch.no_grad():
            for embeddings, label in zip(embeddings_list, labels):
                x = torch.from_numpy(embeddings).float().to(self._device)
                y = torch.tensor([label], dtype=torch.float32).to(self._device)
                
                pred, _ = self._model(x)
                loss = criterion(pred.squeeze(), y.squeeze())
                
                losses.append(loss.item())
                preds.append(pred.item())
        
        avg_loss = np.mean(losses)
        auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
        
        return avg_loss, auc
    
    def predict(self, embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Predict for a single slide.
        
        Args:
            embeddings: Patch embeddings of shape (n_patches, embedding_dim)
            
        Returns:
            Tuple of (probability, attention_weights)
        """
        import torch
        
        if self._model is None:
            self._setup_device()
            self._build_model()
            self._model.to(self._device)
            logger.warning("Using untrained model for prediction")
        
        self._model.eval()
        
        x = torch.from_numpy(embeddings).float().to(self._device)
        
        with torch.no_grad():
            prob, attention = self._model(x, return_attention=True)
        
        return prob.item(), attention.cpu().numpy()
    
    def predict_batch(
        self,
        embeddings_list: List[np.ndarray],
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Predict for multiple slides.
        
        Args:
            embeddings_list: List of embedding arrays
            
        Returns:
            List of (probability, attention_weights) tuples
        """
        results = []
        for embeddings in embeddings_list:
            prob, attention = self.predict(embeddings)
            results.append((prob, attention))
        return results
