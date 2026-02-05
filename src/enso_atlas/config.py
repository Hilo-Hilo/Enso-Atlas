"""
Configuration management for Enso Atlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WSIConfig:
    """WSI processing configuration."""
    patch_size: int = 224
    magnification: int = 20
    tissue_threshold: float = 0.5
    min_tissue_area: float = 0.1
    max_patches_coarse: int = 2000
    max_patches_refine: int = 8000


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "path-foundation"
    batch_size: int = 64
    precision: str = "fp16"
    cache_dir: str = "data/embeddings"


@dataclass
class MILConfig:
    """MIL head configuration."""
    architecture: str = "clam"
    input_dim: int = 384
    hidden_dim: int = 256
    attention_heads: int = 1
    dropout: float = 0.25
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20


@dataclass
class EvidenceConfig:
    """Evidence generation configuration."""
    top_k_patches: int = 12
    heatmap_alpha: float = 0.4
    similarity_k: int = 20
    faiss_index_type: str = "IVF1024,Flat"
    colormap: str = "jet"  # Colormap for heatmap visualization


@dataclass
class ReportingConfig:
    """MedGemma reporting configuration."""
    model: str = "google/medgemma-4b-it"
    max_evidence_patches: int = 8
    max_similar_cases: int = 5
    max_input_tokens: int = 3072
    max_output_tokens: int = 512
    max_generation_time_s: float = 120.0
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class UIConfig:
    """UI configuration."""
    server_port: int = 7860
    share: bool = False
    theme: str = "soft"


@dataclass
class PathsConfig:
    """Path configuration."""
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "outputs"
    model_dir: str = "models"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    offline_mode: bool = True
    log_level: str = "INFO"
    enable_telemetry: bool = False


@dataclass
class AtlasConfig:
    """Master configuration for Enso Atlas."""
    wsi: WSIConfig = field(default_factory=WSIConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AtlasConfig":
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            wsi=WSIConfig(**data.get("wsi", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            mil=MILConfig(**data.get("mil", {})),
            evidence=EvidenceConfig(**data.get("evidence", {})),
            reporting=ReportingConfig(**data.get("reporting", {})),
            ui=UIConfig(**data.get("ui", {})),
            paths=PathsConfig(**data.get("paths", {})),
            deployment=DeploymentConfig(**data.get("deployment", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "wsi": self.wsi.__dict__,
            "embedding": self.embedding.__dict__,
            "mil": self.mil.__dict__,
            "evidence": self.evidence.__dict__,
            "reporting": self.reporting.__dict__,
            "ui": self.ui.__dict__,
            "paths": self.paths.__dict__,
            "deployment": self.deployment.__dict__,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
