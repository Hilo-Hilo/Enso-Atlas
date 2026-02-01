"""
Core orchestration module for Enso Atlas.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import logging

import numpy as np

from .config import AtlasConfig
from .wsi.processor import WSIProcessor
from .embedding.embedder import PathFoundationEmbedder
from .mil.clam import CLAMClassifier
from .evidence.generator import EvidenceGenerator
from .reporting.medgemma import MedGemmaReporter


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of a single slide analysis."""
    slide_path: str
    score: float
    label: str
    confidence: float
    heatmap: np.ndarray
    evidence_patches: List[dict]
    similar_cases: List[dict]
    report: dict
    
    def save_heatmap(self, path: str) -> None:
        """Save heatmap overlay to file."""
        from PIL import Image
        img = Image.fromarray((self.heatmap * 255).astype(np.uint8))
        img.save(path)
    
    def save_report(self, path: str, format: str = "json") -> None:
        """Save report to file."""
        import json
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)


class EnsoAtlas:
    """
    Main orchestrator for Enso Atlas pathology analysis.
    
    Example:
        atlas = EnsoAtlas.from_config("config/default.yaml")
        result = atlas.analyze("path/to/slide.svs")
        print(result.score)
    """
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self._setup_logging()
        
        # Initialize components (lazy loading)
        self._wsi_processor: Optional[WSIProcessor] = None
        self._embedder: Optional[PathFoundationEmbedder] = None
        self._classifier: Optional[CLAMClassifier] = None
        self._evidence_generator: Optional[EvidenceGenerator] = None
        self._reporter: Optional[MedGemmaReporter] = None
        
        logger.info(f"EnsoAtlas initialized with config")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        level = getattr(logging, self.config.deployment.log_level.upper())
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    @classmethod
    def from_config(cls, config_path: str | Path) -> "EnsoAtlas":
        """Create EnsoAtlas from config file."""
        config = AtlasConfig.from_yaml(config_path)
        return cls(config)
    
    @property
    def wsi_processor(self) -> WSIProcessor:
        """Lazy-load WSI processor."""
        if self._wsi_processor is None:
            self._wsi_processor = WSIProcessor(self.config.wsi)
        return self._wsi_processor
    
    @property
    def embedder(self) -> PathFoundationEmbedder:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = PathFoundationEmbedder(self.config.embedding)
        return self._embedder
    
    @property
    def classifier(self) -> CLAMClassifier:
        """Lazy-load MIL classifier."""
        if self._classifier is None:
            self._classifier = CLAMClassifier(self.config.mil)
        return self._classifier
    
    @property
    def evidence_generator(self) -> EvidenceGenerator:
        """Lazy-load evidence generator."""
        if self._evidence_generator is None:
            self._evidence_generator = EvidenceGenerator(self.config.evidence)
        return self._evidence_generator
    
    @property
    def reporter(self) -> MedGemmaReporter:
        """Lazy-load MedGemma reporter."""
        if self._reporter is None:
            self._reporter = MedGemmaReporter(self.config.reporting)
        return self._reporter
    
    def analyze(
        self,
        slide_path: str | Path,
        generate_report: bool = True,
        save_cache: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a single whole-slide image.
        
        Args:
            slide_path: Path to the WSI file
            generate_report: Whether to generate MedGemma report
            save_cache: Whether to cache embeddings
            
        Returns:
            AnalysisResult with score, evidence, and report
        """
        slide_path = Path(slide_path)
        logger.info(f"Analyzing slide: {slide_path.name}")
        
        # Step 1: Extract patches
        logger.info("Extracting patches...")
        patches, coords = self.wsi_processor.extract_patches(slide_path)
        logger.info(f"Extracted {len(patches)} patches")
        
        # Step 2: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed(patches, cache_key=str(slide_path))
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        # Step 3: Run MIL classifier
        logger.info("Running classifier...")
        score, attention_weights = self.classifier.predict(embeddings)
        label = "responder" if score > 0.5 else "non-responder"
        confidence = abs(score - 0.5) * 2  # Convert to 0-1 confidence
        logger.info(f"Prediction: {label} (score={score:.3f})")
        
        # Step 4: Generate evidence
        logger.info("Generating evidence...")
        heatmap = self.evidence_generator.create_heatmap(
            attention_weights, coords, self.wsi_processor.get_slide_dimensions(slide_path)
        )
        evidence_patches = self.evidence_generator.select_top_patches(
            patches, coords, attention_weights, k=self.config.evidence.top_k_patches
        )
        similar_cases = self.evidence_generator.find_similar(
            embeddings, attention_weights, k=self.config.evidence.similarity_k
        )
        
        # Step 5: Generate report (optional)
        report = {}
        if generate_report:
            logger.info("Generating MedGemma report...")
            report = self.reporter.generate(
                evidence_patches=evidence_patches,
                score=score,
                label=label,
                similar_cases=similar_cases,
            )
        
        return AnalysisResult(
            slide_path=str(slide_path),
            score=score,
            label=label,
            confidence=confidence,
            heatmap=heatmap,
            evidence_patches=evidence_patches,
            similar_cases=similar_cases,
            report=report,
        )
    
    def batch_analyze(
        self,
        slide_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.svs",
    ) -> List[AnalysisResult]:
        """
        Analyze all slides in a directory.
        
        Args:
            slide_dir: Directory containing WSI files
            output_dir: Directory to save results
            pattern: Glob pattern for slide files
            
        Returns:
            List of AnalysisResult objects
        """
        slide_dir = Path(slide_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        slides = list(slide_dir.glob(pattern))
        logger.info(f"Found {len(slides)} slides to analyze")
        
        results = []
        for slide_path in slides:
            try:
                result = self.analyze(slide_path)
                
                # Save outputs
                slide_name = slide_path.stem
                result.save_heatmap(output_dir / f"{slide_name}_heatmap.png")
                result.save_report(output_dir / f"{slide_name}_report.json")
                
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {slide_path.name}: {e}")
        
        return results
