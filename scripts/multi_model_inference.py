#!/usr/bin/env python3
"""
Multi-Model Inference for MedGemma Enso Atlas

Loads all trained TransMIL models and runs inference on slide embeddings.
Model configurations are loaded from config/projects.yaml when available,
falling back to built-in defaults.
"""

import json
import sys
import argparse
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transmil import TransMIL


REAL_AUC_RESULTS_HINTS: Dict[str, List[str]] = {
    # Prefixes under data/tcga_full/bucket_training/.
    # We search newest matching directory first.
    "platinum_sensitivity": [
        "results_ov_platinum_api_barcode_v2_balanced_noaug_allbucket_",
        "results_ov_platinum_api_barcode_v2_balanced_noaug_",
        "results_api_expanded_aug_",
        "results_single_split_",
    ],
    "tumor_grade": [
        "results_ov_tumor_grade_api_barcode_v2_balanced_noaug_allbucket_",
        "results_ov_tumor_grade_api_barcode_v2_balanced_noaug_",
        "results_ov_tumor_grade_api_aug_",
    ],
    "survival_1y": [
        "results_ov_survival_1y_api_barcode_v2_balanced_noaug_allbucket_",
        "results_ov_survival_1y_api_barcode_v2_balanced_noaug_",
        "results_ov_survival_1y_api_aug_",
    ],
    "survival_3y": [
        "results_ov_survival_3y_api_barcode_v2_balanced_noaug_allbucket_",
        "results_ov_survival_3y_api_barcode_v2_balanced_noaug_",
        "results_ov_survival_3y_api_aug_",
    ],
    "survival_5y": [
        "results_ov_survival_5y_api_barcode_v2_balanced_noaug_allbucket_",
        "results_ov_survival_5y_api_barcode_v2_balanced_noaug_",
        "results_ov_survival_5y_api_aug_",
    ],
    "lung_stage": [
        "results_luad_stage_api_barcode_v2_balanced_noaug_allbucket_",
        "results_luad_stage_api_barcode_v2_balanced_noaug_",
        "results_luad_stage_api_aug_",
    ],
}

DEFAULT_DECISION_THRESHOLDS: Dict[str, float] = {
    "platinum_sensitivity": 0.5,
    # New retrained weights default to midpoint threshold unless re-optimized.
    "tumor_grade": 0.5,
    "survival_1y": 0.5,
    "survival_3y": 0.5,
    "survival_5y": 0.5,
    "lung_stage": 0.5,
}


def _sanitize_threshold(value: Any, fallback: float = 0.5) -> float:
    """Clamp decision threshold to a safe open interval (0, 1)."""
    try:
        t = float(value)
    except Exception:
        t = float(fallback)

    if not np.isfinite(t):
        t = float(fallback)

    # Keep away from 0/1 to avoid divide-by-zero in confidence scaling.
    return float(min(0.99, max(0.01, t)))


def _resolve_decision_threshold(model_id: str, cfg: Dict[str, Any]) -> float:
    """
    Resolve decision threshold with precedence:
      1) env MULTIMODEL_THRESHOLD_<MODEL_ID>
      2) config decision_threshold (or legacy threshold)
      3) model-specific default
    """
    fallback = _sanitize_threshold(DEFAULT_DECISION_THRESHOLDS.get(model_id, 0.5), 0.5)

    cfg_threshold = cfg.get("decision_threshold", cfg.get("threshold", fallback))
    resolved = _sanitize_threshold(cfg_threshold, fallback)

    env_key = f"MULTIMODEL_THRESHOLD_{model_id.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None and str(env_val).strip() != "":
        resolved = _sanitize_threshold(env_val, resolved)

    return resolved


def _extract_auc_from_results_json(path: Path) -> Optional[float]:
    """Extract scalar AUC from a training results.json file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    auc_val = None
    try:
        auc_block = (payload.get("aggregate_metrics") or {}).get("auc")
        if isinstance(auc_block, dict):
            auc_val = auc_block.get("mean")
        if auc_val is None:
            auc_val = payload.get("best_fold_auc")
        if auc_val is None:
            return None
        auc = float(auc_val)
        if 0.0 <= auc <= 1.0:
            return auc
    except Exception:
        return None
    return None


def _resolve_real_auc_for_model(model_id: str) -> Optional[float]:
    """
    Resolve latest real AUC for a model from local training results files.
    Falls back to YAML/default AUC when no matching results are found.
    """
    hints = REAL_AUC_RESULTS_HINTS.get(model_id, [])
    if not hints:
        return None

    bucket_training_dir = PROJECT_ROOT / "data" / "tcga_full" / "bucket_training"
    if not bucket_training_dir.exists():
        return None

    for prefix in hints:
        pattern = f"{prefix}*/results.json"
        candidates = list(bucket_training_dir.glob(pattern))
        if not candidates:
            continue

        candidates.sort(
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )

        for p in candidates:
            auc = _extract_auc_from_results_json(p)
            if auc is not None:
                return auc

    return None


def _load_model_configs_from_yaml() -> Optional[Dict[str, Dict]]:
    """Try to load model configs from projects.yaml."""
    try:
        import yaml
        yaml_path = PROJECT_ROOT / "config" / "projects.yaml"
        if not yaml_path.exists():
            return None
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        models = cfg.get("classification_models", {})
        if not models:
            return None
        resolved = {}
        for mid, m in models.items():
            auc = m.get("auc", 0.5)
            real_auc = _resolve_real_auc_for_model(mid)
            if real_auc is not None:
                auc = real_auc
            resolved[mid] = {
                "model_dir": m.get("model_dir", mid),
                "display_name": m.get("display_name", mid),
                "description": m.get("description", ""),
                "auc": auc,
                "n_slides": m.get("n_slides", 0),
                "category": m.get("category", "general_pathology"),
                "positive_label": m.get("positive_label", "Positive"),
                "negative_label": m.get("negative_label", "Negative"),
            }
        return resolved
    except Exception as e:
        print(f"Warning: Could not load YAML config: {e}")
        return None


# Model configurations - loaded from YAML if available, otherwise defaults
_YAML_CONFIGS = _load_model_configs_from_yaml()

# Built-in fallback configurations
_DEFAULT_CONFIGS = {
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

# Use YAML configs if loaded, otherwise fall back to defaults
MODEL_CONFIGS = _YAML_CONFIGS if _YAML_CONFIGS else _DEFAULT_CONFIGS
for _model_id, _cfg in MODEL_CONFIGS.items():
    _cfg["decision_threshold"] = _resolve_decision_threshold(_model_id, _cfg)

# Temperature scaling per model (> 1.0 softens extreme predictions)
# Applied as: logit / temperature -> sigmoid
# This corrects for uncalibrated training on imbalanced datasets
MODEL_TEMPERATURES = {
    "platinum_sensitivity": 1.0,  # Well-calibrated (AUC 0.91)
    "tumor_grade": 1.0,          # Reasonably calibrated (AUC 0.75)
    "survival_5y": 2.5,          # Very extreme outputs, needs heavy smoothing
    "survival_3y": 2.0,          # Extreme outputs
    "survival_1y": 2.5,          # Very extreme outputs
    "lung_stage": 1.0,           # New model, default temperature
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

    @staticmethod
    def _uniform_subsample_indices(total: int, target: int) -> np.ndarray:
        if target >= total:
            return np.arange(total, dtype=np.int64)
        idx = np.linspace(0, total - 1, num=target, dtype=np.int64)
        return np.unique(idx)

    @staticmethod
    def _release_cuda_cache() -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _resolve_min_attention_patches() -> int:
        raw = str(os.environ.get("MULTIMODEL_MIN_PATCHES_FOR_ATTENTION", "128")).strip()
        try:
            val = int(raw)
        except ValueError:
            val = 128
        return max(16, val)
    
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
        
        # Load weights.
        # Support both:
        #   1) plain state_dict checkpoints
        #   2) wrapped checkpoints from train_transmil_finetune.py with:
        #      {"model_state_dict": ..., "config": ..., ...}
        # Newer PyTorch "weights_only=True" can fail on wrapped checkpoints
        # that contain numpy scalar metadata, so we fall back to weights_only=False
        # only when needed.
        ckpt_cfg = {}
        try:
            loaded = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
                maybe_cfg = loaded.get("config", {})
                if isinstance(maybe_cfg, dict):
                    ckpt_cfg = maybe_cfg
            elif isinstance(loaded, dict):
                state_dict = loaded
            else:
                raise TypeError(f"Unexpected checkpoint type with weights_only=True: {type(loaded)!r}")
        except Exception:
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
                maybe_cfg = loaded.get("config", {})
                if isinstance(maybe_cfg, dict):
                    ckpt_cfg = maybe_cfg
            elif isinstance(loaded, dict):
                state_dict = loaded
            else:
                raise TypeError(f"Unsupported checkpoint format at {model_path}: {type(loaded)!r}")
            print(
                "Warning: loaded wrapped checkpoint with weights_only=False. "
                "Use only trusted model files."
            )

        if ckpt_cfg:
            model_config.update({k: ckpt_cfg[k] for k in model_config if k in ckpt_cfg})

        # Create and load model
        model = TransMIL(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )

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
                "decision_threshold": config.get("decision_threshold", 0.5),
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

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            return {"error": f"Invalid embeddings shape: {emb.shape}"}
        total_patches = int(emb.shape[0])
        if total_patches == 0:
            return {"error": "Empty embedding bag"}

        min_patches = self._resolve_min_attention_patches()
        indices = np.arange(total_patches, dtype=np.int64)

        while True:
            try:
                x = torch.from_numpy(emb[indices]).float().to(self.device)
                if return_attention:
                    score_t, attention_t = model(x, return_attention=True)
                    sampled_attention = attention_t.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    score_t = model(x)
                    sampled_attention = None
                score = score_t.detach().cpu().item()
                break
            except torch.cuda.OutOfMemoryError as e:
                if self.device != "cuda":
                    raise

                current = int(indices.size)
                if current <= min_patches:
                    return {
                        "error": (
                            f"CUDA OOM at minimum sampled patches ({min_patches}) for {model_id}. "
                            "Reduce MULTIMODEL_MIN_PATCHES_FOR_ATTENTION."
                        )
                    }

                new_size = max(min_patches, current // 2)
                if new_size >= current:
                    return {
                        "error": (
                            f"CUDA OOM and unable to reduce sampled patches below {current} for {model_id}."
                        )
                    }

                print(
                    f"CUDA OOM for {model_id} at {current} patches; retrying on GPU with {new_size} patches."
                )
                self._release_cuda_cache()
                indices = self._uniform_subsample_indices(total_patches, new_size)
            finally:
                try:
                    del x
                except Exception:
                    pass
                self._release_cuda_cache()
        
        # Apply temperature scaling to soften extreme predictions
        # Convert sigmoid output -> logit -> scale -> sigmoid
        temperature = MODEL_TEMPERATURES.get(model_id, 1.0)
        if temperature != 1.0 and 0.0 < score < 1.0:
            import math
            logit = math.log(score / (1.0 - score))  # inverse sigmoid
            scaled_logit = logit / temperature
            score = 1.0 / (1.0 + math.exp(-scaled_logit))  # sigmoid

        score = float(min(1.0, max(0.0, score)))
        decision_threshold = _sanitize_threshold(config.get("decision_threshold", 0.5), 0.5)

        # Determine label based on per-model threshold
        is_positive = score >= decision_threshold
        label = config["positive_label"] if is_positive else config["negative_label"]
        if is_positive:
            denom = max(1e-6, 1.0 - decision_threshold)
            confidence = (score - decision_threshold) / denom
        else:
            denom = max(1e-6, decision_threshold)
            confidence = (decision_threshold - score) / denom
        confidence = float(min(max(confidence, 0.0), 0.99))
        
        result = {
            "model_id": model_id,
            "model_name": config["display_name"],
            "category": config["category"],
            "score": score,
            "decision_threshold": decision_threshold,
            "label": label,
            "positive_label": config["positive_label"],
            "negative_label": config["negative_label"],
            "confidence": confidence,
            "auc": config["auc"],
            "n_training_slides": config["n_slides"],
            "description": config["description"],
        }
        
        if sampled_attention is not None:
            if indices.size == total_patches:
                attention = sampled_attention
            else:
                attention = np.zeros(total_patches, dtype=np.float32)
                attention[indices] = sampled_attention
                s = float(attention.sum())
                if s > 0.0:
                    attention /= s
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
        
        # Survival consistency check: if 1yr says deceased, 3yr/5yr cannot
        # logically say survived. Add warnings for contradictory predictions.
        warnings = []
        surv_1y = predictions.get("survival_1y", {})
        surv_3y = predictions.get("survival_3y", {})
        surv_5y = predictions.get("survival_5y", {})

        if surv_1y.get("label") and surv_3y.get("label") and surv_5y.get("label"):
            s1_deceased = surv_1y["label"] == "Deceased"
            s3_survived = surv_3y["label"] == "Survived"
            s5_survived = surv_5y["label"] == "Survived"

            if s1_deceased and (s3_survived or s5_survived):
                warnings.append(
                    "Survival predictions are contradictory: 1-year predicts Deceased "
                    f"(score {surv_1y.get('score', 0):.3f}) while "
                    + ("3-year" if s3_survived else "")
                    + (" and " if s3_survived and s5_survived else "")
                    + ("5-year" if s5_survived else "")
                    + " predict Survived. Treat survival estimates with caution."
                )
                # Add per-model warning fields
                if s3_survived:
                    predictions["survival_3y"]["warning"] = (
                        "Contradicts 1-year survival prediction (Deceased). Interpret with caution."
                    )
                if s5_survived:
                    predictions["survival_5y"]["warning"] = (
                        "Contradicts 1-year survival prediction (Deceased). Interpret with caution."
                    )

            # Also check: if 3yr says deceased but 5yr says survived
            if surv_3y["label"] == "Deceased" and s5_survived:
                warnings.append(
                    "3-year survival predicts Deceased but 5-year predicts Survived. "
                    "This is logically inconsistent. Treat survival estimates with caution."
                )
                predictions["survival_5y"]["warning"] = (
                    "Contradicts 3-year survival prediction (Deceased). Interpret with caution."
                )

        # Group by category (dynamic -- any category key from config)
        by_category: dict = {}
        for k, pred in predictions.items():
            cat = pred.get("category", "general_pathology")
            by_category.setdefault(cat, []).append(pred)
        
        result = {
            "predictions": predictions,
            "by_category": by_category,
            "n_patches": len(embeddings),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        }

        if warnings:
            result["warnings"] = warnings

        return result


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
