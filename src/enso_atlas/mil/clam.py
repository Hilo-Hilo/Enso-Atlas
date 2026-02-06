"""
CLAM - Clustering-constrained Attention Multiple Instance Learning.

Implementation based on:
"Data-efficient and weakly supervised computational pathology on whole-slide images"
Lu et al., Nature Biomedical Engineering, 2021

This module provides the attention-based MIL head for slide-level classification
with interpretable attention weights for evidence generation.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

import numpy as np

from enso_atlas.config import MILConfig

logger = logging.getLogger(__name__)



import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyCLAMModel(nn.Module):
    """Simpler CLAM model compatible with older checkpoints.
    
    Architecture matches the trained model:
    - encoder: Linear(384, 256) + ReLU + Dropout
    - attention: Gated attention (attention_a, attention_b, attention_c)
    - classifier: Linear(256, 2)
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Gated attention mechanism
        self.attention = nn.ModuleDict({
            "attention_a": nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh()),
            "attention_b": nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid()),
            "attention_c": nn.Linear(attention_dim, 1)
        })
        
        # Bag classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x, return_attention: bool = True):
        """Forward pass."""
        # Encode features
        h = self.encoder(x)  # (n_patches, hidden_dim)
        
        # Compute gated attention
        a = self.attention["attention_a"](h)
        b = self.attention["attention_b"](h)
        A = self.attention["attention_c"](a * b)
        A = F.softmax(A, dim=0)
        
        # Aggregate with attention
        M = torch.mm(A.T, h)  # (1, hidden_dim)
        
        # Classify
        logits = self.classifier(M)
        score = F.softmax(logits, dim=1)[0, 1]
        
        if return_attention:
            return score, A.squeeze(-1)
        return score


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

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        # Handle both wrapped checkpoints and plain state_dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Detect legacy checkpoint format by checking for "encoder" key
        is_legacy = any(k.startswith("encoder.") for k in state_dict.keys())
        
        if is_legacy:
            # Use legacy model architecture
            logger.info("Detected legacy checkpoint format, using LegacyCLAMModel")
            self._model = LegacyCLAMModel(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim
            ).to(self._device)
            
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

    def predict_with_uncertainty(
        self,
        embeddings: np.ndarray,
        n_samples: int = 20,
    ) -> Dict[str, float | np.ndarray]:
        """
        Predict with MC Dropout uncertainty quantification.

        Runs multiple forward passes with dropout enabled to estimate
        predictive uncertainty. High variance indicates model uncertainty.

        Args:
            embeddings: Patch embeddings of shape (n_patches, embedding_dim)
            n_samples: Number of Monte Carlo forward passes (default 20)

        Returns:
            Dictionary containing:
            - prediction: "RESPONDER" or "NON-RESPONDER"
            - probability: Mean predicted probability
            - uncertainty: Standard deviation of predictions
            - confidence_interval: [lower, upper] 95% CI bounds
            - attention_weights: Mean attention weights
            - attention_uncertainty: Std of attention weights
            - samples: Raw probability samples (for visualization)
            - is_uncertain: Boolean flag if uncertainty exceeds threshold
        """
        import torch

        if self._model is None:
            self._setup_device()
            self._build_model()
            self._model.to(self._device)
            logger.warning("Using untrained model for prediction")

        # Enable dropout for MC sampling by setting to train mode
        # but we still don't compute gradients
        self._model.train()

        x = torch.from_numpy(embeddings).float().to(self._device)

        predictions = []
        attention_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                prob, attention = self._model(x, return_attention=True)
                predictions.append(prob.item())
                attention_samples.append(attention.cpu().numpy())

        # Set back to eval mode
        self._model.eval()

        # Compute statistics
        predictions = np.array(predictions)
        attention_samples = np.array(attention_samples)  # (n_samples, n_patches)

        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))

        # 95% confidence interval (approximately 2 standard deviations)
        ci_lower = float(max(0.0, mean_pred - 2 * std_pred))
        ci_upper = float(min(1.0, mean_pred + 2 * std_pred))

        # Mean attention weights and their uncertainty
        mean_attention = np.mean(attention_samples, axis=0)
        attention_std = np.std(attention_samples, axis=0)

        # Uncertainty threshold for flagging cases
        # Threshold chosen based on typical MC Dropout variance
        UNCERTAINTY_THRESHOLD = 0.15
        is_uncertain = std_pred > UNCERTAINTY_THRESHOLD

        return {
            "prediction": "RESPONDER" if mean_pred > 0.5 else "NON-RESPONDER",
            "probability": mean_pred,
            "uncertainty": std_pred,
            "confidence_interval": [ci_lower, ci_upper],
            "attention_weights": mean_attention,
            "attention_uncertainty": attention_std,
            "samples": predictions.tolist(),
            "is_uncertain": is_uncertain,
            "n_samples": n_samples,
        }


# ---------------------------------------------------------------------------
# TransMIL Classifier -- drop-in alternative to CLAMClassifier
# ---------------------------------------------------------------------------

class TransMILClassifier:
    """
    TransMIL (Transformer-based MIL) classifier.

    Provides the same public interface as CLAMClassifier so the two can be
    swapped transparently:

        classifier.load(path)
        score, attention = classifier.predict(embeddings)

    The underlying model is imported from ``models/transmil.py``.
    """

    def __init__(self, config: MILConfig):
        self.config = config
        self._model = None
        self._device = None
        self._is_trained = False

    def _setup_device(self):
        """Select the best available device."""
        import torch

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        return self._device

    def _build_model(self, cfg_override: Optional[Dict] = None):
        """
        Instantiate the TransMIL network.

        Args:
            cfg_override: Optional dict of constructor kwargs that takes
                precedence over ``self.config`` (used when loading a
                checkpoint that stores its own config).
        """
        # Lazy import so the rest of the module does not require the models
        # package to be on sys.path at import time.
        import importlib
        import sys
        from pathlib import Path as _Path

        # Ensure the models/ directory is importable.
        models_dir = str(_Path(__file__).resolve().parents[3] / "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        from transmil import TransMIL  # type: ignore[import-untyped]

        c = cfg_override or {}
        self._model = TransMIL(
            input_dim=c.get("input_dim", self.config.input_dim),
            hidden_dim=c.get("hidden_dim", getattr(self.config, "hidden_dim", 512)),
            num_classes=c.get("num_classes", 1),
            num_heads=c.get("num_heads", getattr(self.config, "attention_heads", 8)),
            num_layers=c.get("num_layers", 2),
            dropout=c.get("dropout", self.config.dropout),
        )
        return self._model

    # ----- public API (mirrors CLAMClassifier) -----

    def load(self, path: str | Path) -> None:
        """
        Load a trained TransMIL checkpoint from disk.

        Supports both wrapped checkpoints (with ``model_state_dict`` key)
        and plain state-dicts.
        """
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TransMIL checkpoint not found: {path}")

        self._setup_device()

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            cfg = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            cfg = {}

        self._build_model(cfg_override=cfg)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._is_trained = True

        logger.info("Loaded TransMIL model from %s", path)

    def save(self, path: str | Path) -> None:
        """Save the model state dict to *path*."""
        import torch

        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("Saved TransMIL model to %s", path)

    def predict(self, embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Predict for a single slide.

        Args:
            embeddings: Patch embeddings of shape ``(n_patches, embedding_dim)``.

        Returns:
            ``(probability, attention_weights)`` where *probability* is a
            float in [0, 1] and *attention_weights* is a 1-D array of
            length ``n_patches``.
        """
        import torch

        if self._model is None:
            self._setup_device()
            self._build_model()
            self._model.to(self._device)
            logger.warning("Using untrained TransMIL model for prediction")

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
            embeddings_list: List of per-slide embedding arrays.

        Returns:
            List of ``(probability, attention_weights)`` tuples.
        """
        return [self.predict(emb) for emb in embeddings_list]

    def predict_with_uncertainty(
        self,
        embeddings: np.ndarray,
        n_samples: int = 20,
    ) -> Dict[str, float | np.ndarray]:
        """
        MC-Dropout uncertainty estimation (same semantics as CLAMClassifier).

        Runs *n_samples* stochastic forward passes with dropout enabled
        and returns summary statistics.
        """
        import torch

        if self._model is None:
            self._setup_device()
            self._build_model()
            self._model.to(self._device)
            logger.warning("Using untrained TransMIL model for prediction")

        self._model.train()  # enable dropout

        x = torch.from_numpy(embeddings).float().to(self._device)

        predictions = []
        attention_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                prob, attention = self._model(x, return_attention=True)
                predictions.append(prob.item())
                attention_samples.append(attention.cpu().numpy())

        self._model.eval()

        predictions_arr = np.array(predictions)
        attention_arr = np.array(attention_samples)

        mean_pred = float(np.mean(predictions_arr))
        std_pred = float(np.std(predictions_arr))

        ci_lower = float(max(0.0, mean_pred - 2 * std_pred))
        ci_upper = float(min(1.0, mean_pred + 2 * std_pred))

        mean_attention = np.mean(attention_arr, axis=0)
        attention_std = np.std(attention_arr, axis=0)

        UNCERTAINTY_THRESHOLD = 0.15

        return {
            "prediction": "RESPONDER" if mean_pred > 0.5 else "NON-RESPONDER",
            "probability": mean_pred,
            "uncertainty": std_pred,
            "confidence_interval": [ci_lower, ci_upper],
            "attention_weights": mean_attention,
            "attention_uncertainty": attention_std,
            "samples": predictions_arr.tolist(),
            "is_uncertain": std_pred > UNCERTAINTY_THRESHOLD,
            "n_samples": n_samples,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_classifier(config: MILConfig) -> CLAMClassifier | TransMILClassifier:
    """
    Instantiate the appropriate MIL classifier based on ``config.architecture``.

    Supported values for ``config.architecture``:
      - ``"clam"`` (default) -- CLAMClassifier
      - ``"transmil"``       -- TransMILClassifier
    """
    arch = getattr(config, "architecture", "clam").lower()
    if arch == "transmil":
        return TransMILClassifier(config)
    return CLAMClassifier(config)
