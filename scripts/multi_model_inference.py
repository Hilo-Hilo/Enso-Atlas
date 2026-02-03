#!/usr/bin/env python3
"""
Multi-Model Inference for MedGemma Enso Atlas

Loads all trained TransMIL models and runs inference on slide embeddings.

Models available:
- transmil_v2: Platinum Sensitivity (AUC 0.907)
- transmil_grade: Tumor Grade High/Low (AUC 0.752)
- transmil_surv5y: 5-Year Survival (AUC 0.697)
- transmil_full: 3-Year Survival (AUC 0.645)
- transmil_surv1y: 1-Year Survival (AUC 0.639)
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transmil import TransMIL


# Model configurations with metadata
MODEL_CONFIGS = {
    "platinum_sensitivity": {
        "model_dir": "transmil_v2",
        "display_name": "Platinum Sensitivity",
        "description": "Predicts response to platinum-based chemotherapy",
        "auc": 0.907,
        "n_slides": 199,
        "category": "ovarian_cancer",
        "positive_label": "Sensitive",
        "negative_label": "Resistant",
    },
    "tumor_grade": {
        "model_dir": "transmil_grade",
        "display_name": "Tumor Grade",
        "description": "Predicts tumor grade (high vs low grade)",
        "auc": 0.752,
        "n_slides": 918,
        "category": "general_pathology",
        "positive_label": "High Grade",
        "negative_label": "Low Grade",
    },
    "survival_5y": {
        "model_dir": "transmil_surv5y",
        "display_name": "5-Year Survival",
        "description": "Predicts 5-year overall survival probability",
        "auc": 0.697,
        "n_slides": 965,
        "category": "ovarian_cancer",
        "positive_label": "Survived",
        "negative_label": "Deceased",
    },
    "survival_3y": {
        "model_dir": "transmil_full",
        "display_name": "3-Year Survival",
        "description": "Predicts 3-year overall survival probability",
        "auc": 0.645,
        "n_slides": 1106,
        "category": "ovarian_cancer",
        "positive_label": "Survived",
        "negative_label": "Deceased",
    },
    "survival_1y": {
        "model_dir": "transmil_surv1y",
        "display_name": "1-Year Survival",
        "description": "Predicts 1-year overall survival probability",
        "auc": 0.639,
        "n_slides": 1135,
        "category": "ovarian_cancer",
        "positive_label": "Survived",
        "negative_label": "Deceased",
    },
}


class MultiModelInference:
    """Handles inference across multiple TransMIL models."""
    
    def __init__(
        self,
        models_dir: Path,
        device: str = "auto",
        load_all: bool = True,
    ):
        """
        Initialize multi-model inference.
        
        Args:
            models_dir: Directory containing model subdirectories (transmil_v2, etc.)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            load_all: If True, load all models immediately. Otherwise, lazy load.
        """
        self.models_dir = Path(models_dir)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Model cache
        self.models: Dict[str, TransMIL] = {}
        self.model_configs: Dict[str, Dict] = {}
        
        if load_all:
            self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models."""
        for model_id, config in MODEL_CONFIGS.items():
            self._load_model(model_id)
    
    def _load_model(self, model_id: str) -> Optional[TransMIL]:
        """Load a single model by ID."""
        if model_id in self.models:
            return self.models[model_id]
        
        if model_id not in MODEL_CONFIGS:
            print(f"Unknown model: {model_id}")
            return None
        
        config = MODEL_CONFIGS[model_id]
        model_dir = self.models_dir / config["model_dir"]
        model_path = model_dir / "best_model.pt"
        config_path = model_dir / "config.json"
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None
        
        # Load model config
        model_config = {
            "input_dim": 384,
            "hidden_dim": 512,
            "num_classes": 1,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.25,
        }
        
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
                model_config.update({
                    k: v for k, v in saved_config.items() 
                    if k in model_config
                })
        
        # Create and load model
        model = TransMIL(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        self.models[model_id] = model
        self.model_configs[model_id] = {**config, **model_config}
        
        print(f"Loaded model: {config['display_name']} (AUC: {config['auc']:.3f})")
        return model
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with metadata."""
        available = []
        for model_id, config in MODEL_CONFIGS.items():
            model_dir = self.models_dir / config["model_dir"]
            model_path = model_dir / "best_model.pt"
            
            available.append({
                "id": model_id,
                "name": config["display_name"],
                "description": config["description"],
                "auc": config["auc"],
                "n_slides": config["n_slides"],
                "category": config["category"],
                "positive_label": config["positive_label"],
                "negative_label": config["negative_label"],
                "available": model_path.exists(),
            })
        
        return available
    
    @torch.no_grad()
    def predict_single(
        self,
        embeddings: np.ndarray,
        model_id: str,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Run prediction with a single model.
        
        Args:
            embeddings: (N, dim) patch embeddings
            model_id: ID of model to use
            return_attention: If True, return attention weights
            
        Returns:
            Dictionary with prediction results
        """
        model = self._load_model(model_id)
        if model is None:
            return {"error": f"Model {model_id} not available"}
        
        config = MODEL_CONFIGS[model_id]
        
        # Convert to tensor
        x = torch.from_numpy(embeddings).float().to(self.device)
        
        # Run inference
        if return_attention:
            score, attention = model(x, return_attention=True)
            attention = attention.cpu().numpy()
        else:
            score = model(x)
            attention = None
        
        score = score.cpu().item()
        
        # Determine label based on threshold
        is_positive = score >= 0.5
        label = config["positive_label"] if is_positive else config["negative_label"]
        confidence = abs(score - 0.5) * 2  # Rescale to 0-1
        
        result = {
            "model_id": model_id,
            "model_name": config["display_name"],
            "category": config["category"],
            "score": score,
            "label": label,
            "positive_label": config["positive_label"],
            "negative_label": config["negative_label"],
            "confidence": confidence,
            "auc": config["auc"],
            "n_training_slides": config["n_slides"],
            "description": config["description"],
        }
        
        if attention is not None:
            result["attention"] = attention.tolist()
        
        return result
    
    @torch.no_grad()
    def predict_all(
        self,
        embeddings: np.ndarray,
        model_ids: Optional[List[str]] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Run prediction with all (or specified) models.
        
        Args:
            embeddings: (N, dim) patch embeddings
            model_ids: List of model IDs to use (None = all)
            return_attention: If True, return attention weights
            
        Returns:
            Dictionary with all prediction results
        """
        if model_ids is None:
            model_ids = list(MODEL_CONFIGS.keys())
        
        predictions = {}
        for model_id in model_ids:
            result = self.predict_single(
                embeddings, 
                model_id, 
                return_attention=return_attention
            )
            predictions[model_id] = result
        
        # Group by category
        ovarian_cancer = [
            predictions[k] for k in predictions 
            if predictions[k].get("category") == "ovarian_cancer"
        ]
        general_pathology = [
            predictions[k] for k in predictions 
            if predictions[k].get("category") == "general_pathology"
        ]
        
        return {
            "predictions": predictions,
            "by_category": {
                "ovarian_cancer": ovarian_cancer,
                "general_pathology": general_pathology,
            },
            "n_patches": len(embeddings),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-model TransMIL inference")
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        required=False,
        help="Path to embeddings .npy file"
    )
    parser.add_argument(
        "--models-dir", "-m",
        type=Path,
        default=Path.home() / "med-gemma-hackathon" / "outputs",
        help="Directory containing model outputs"
    )
    parser.add_argument(
        "--models", 
        nargs="+",
        default=None,
        help="Specific models to run (default: all)"
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Return attention weights"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = MultiModelInference(
        models_dir=args.models_dir,
        device=args.device,
        load_all=not args.list_models,
    )
    
    # List models mode
    if args.list_models:
        models = inference.get_available_models()
        print(json.dumps(models, indent=2))
        return
    
    # Check embeddings argument
    if args.embeddings is None:
        print("Error: --embeddings is required for inference", file=sys.stderr)
        sys.exit(1)
    
    # Load embeddings
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)
    
    embeddings = np.load(args.embeddings)
    print(f"Loaded embeddings: {embeddings.shape}", file=sys.stderr)
    
    # Run inference
    results = inference.predict_all(
        embeddings,
        model_ids=args.models,
        return_attention=args.attention,
    )
    
    # Output
    output_json = json.dumps(results, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
