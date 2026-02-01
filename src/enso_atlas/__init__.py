"""
Enso Atlas - On-Prem Pathology Evidence Engine for Treatment-Response Insight

A local-first, evidence-based pathology analysis system using:
- Path Foundation for patch embeddings
- CLAM for attention-based MIL
- FAISS for similarity search
- MedGemma for structured reporting
"""

__version__ = "0.1.0"

from .core import EnsoAtlas
from .config import AtlasConfig

__all__ = ["EnsoAtlas", "AtlasConfig", "__version__"]
