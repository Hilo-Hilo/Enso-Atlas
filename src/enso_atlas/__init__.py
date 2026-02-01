"""
Enso Atlas - On-Prem Pathology Evidence Engine for Treatment-Response Insight

A local-first, evidence-based pathology analysis system using:
- Path Foundation for patch embeddings
- CLAM for attention-based MIL
- FAISS for similarity search
- MedGemma for structured reporting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .config import AtlasConfig
    from .core import EnsoAtlas

__all__ = ["EnsoAtlas", "AtlasConfig", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "EnsoAtlas":
        from .core import EnsoAtlas as _EnsoAtlas

        return _EnsoAtlas
    if name == "AtlasConfig":
        from .config import AtlasConfig as _AtlasConfig

        return _AtlasConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
