"""
Clinical Decision Support Module

Provides actionable clinical guidance based on prediction results,
confidence levels, slide quality, and similar case outcomes.

NOTE: This is a RESEARCH TOOL. All recommendations require validation
by qualified clinicians and should not replace clinical judgment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Model confidence categorization."""
    HIGH = "high"      # >= 0.8
    MODERATE = "moderate"  # 0.6 - 0.8
    LOW = "low"        # < 0.6


class RiskLevel(Enum):
    """Overall risk stratification level."""
    HIGH_CONFIDENCE = "high_confidence"
    MODERATE_CONFIDENCE = "moderate_confidence"
    LOW_CONFIDENCE = "low_confidence"
    INCONCLUSIVE = "inconclusive"


@dataclass
class QualityFactors:
    """Quality factors that affect recommendation confidence."""
    slide_quality: str = "unknown"  # good, acceptable, poor
    tissue_coverage: Optional[float] = None
    blur_score: Optional[float] = None
    artifact_detected: bool = False
    
    @property
    def quality_concerns(self) -> List[str]:
        """List quality issues that may affect interpretation."""
        concerns = []
        if self.slide_quality == "poor":
            concerns.append("Poor overall slide quality")
        if self.tissue_coverage is not None and self.tissue_coverage < 0.5:
            concerns.append(f"Low tissue coverage ({self.tissue_coverage:.0%})")
        if self.blur_score is not None and self.blur_score > 0.2:
            concerns.append(f"Significant blur detected")
        if self.artifact_detected:
            concerns.append("Processing artifacts present")
        return concerns
    
    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable for reliable predictions."""
        if self.slide_quality == "poor":
            return False
        if self.tissue_coverage is not None and self.tissue_coverage < 0.4:
            return False
        if self.blur_score is not None and self.blur_score > 0.3:
            return False
        return True


@dataclass 
class SimilarCaseOutcomes:
    """Aggregated outcomes from similar cases."""
    total_similar: int = 0
    responders: int = 0
    non_responders: int = 0
    unknown: int = 0
    avg_similarity: float = 0.0
    
    @property
    def responder_ratio(self) -> Optional[float]:
        """Ratio of responders among similar cases with known outcomes."""
        known = self.responders + self.non_responders
        if known == 0:
            return None
        return self.responders / known
    
    @property
    def has_sufficient_evidence(self) -> bool:
        """Check if we have enough similar cases for comparison."""
        return (self.responders + self.non_responders) >= 2


@dataclass
class DecisionSupportOutput:
    """Structured clinical decision support output."""
    # Risk stratification
    risk_level: RiskLevel
    confidence_level: ConfidenceLevel
    confidence_score: float
    
    # Recommendations
    primary_recommendation: str
    supporting_rationale: List[str]
    alternative_considerations: List[str]
    
    # Clinical guidelines references
    guideline_references: List[Dict[str, str]]
    
    # Uncertainty messaging
    uncertainty_statement: str
    quality_warnings: List[str]
    
    # Suggested actions
    suggested_workup: List[str]
    
    # Interpretation aids
    interpretation_note: str
    caveat: str = "This is a research decision-support tool only. All findings must be validated by qualified pathologists and oncologists."


class ClinicalDecisionSupport:
    """
    Clinical decision support engine for treatment response prediction.
    
    Generates actionable guidance based on:
    - Model prediction and confidence
    - Slide quality metrics
    - Similar case outcomes
    - Cancer-type-specific clinical guidelines
    """
    
    # NCCN guideline references by cancer type
    NCCN_GUIDELINES_BY_CANCER = {
        "ovarian_cancer": {
            "positive": {
                "source": "NCCN Guidelines for Ovarian Cancer",
                "section": "Recurrent Disease - Platinum-Sensitive",
                "recommendation": "Consider platinum-based combination therapy for platinum-sensitive recurrence (>6 months since last platinum therapy)",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1453"
            },
            "negative": {
                "source": "NCCN Guidelines for Ovarian Cancer", 
                "section": "Recurrent Disease - Platinum-Resistant",
                "recommendation": "Consider non-platinum single agents, targeted therapy, or clinical trial enrollment for platinum-resistant disease (<6 months)",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1453"
            },
            "molecular_testing": {
                "source": "NCCN Guidelines for Ovarian Cancer",
                "section": "Principles of Pathology Review",
                "recommendation": "Germline and somatic testing for BRCA1/2, HRD status recommended for treatment planning",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1453"
            },
            "targeted_therapy": {
                "source": "NCCN Guidelines for Ovarian Cancer",
                "section": "Systemic Therapy for Recurrent Disease",
                "recommendation": "Bevacizumab may be added to chemotherapy in platinum-sensitive or platinum-resistant settings",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1453"
            }
        },
        "lung_cancer": {
            "positive": {
                "source": "NCCN Guidelines for Non-Small Cell Lung Cancer",
                "section": "Treatment of Stage III/IV Disease",
                "recommendation": "Consider systemic therapy based on stage, molecular profile, and performance status",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450"
            },
            "negative": {
                "source": "NCCN Guidelines for Non-Small Cell Lung Cancer",
                "section": "Treatment of Stage I/II Disease",
                "recommendation": "Consider surgical resection with or without adjuvant therapy for early-stage disease",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450"
            },
            "molecular_testing": {
                "source": "NCCN Guidelines for Non-Small Cell Lung Cancer",
                "section": "Principles of Molecular and Biomarker Analysis",
                "recommendation": "Broad molecular profiling (EGFR, ALK, ROS1, BRAF, KRAS, PD-L1) recommended for advanced disease",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450"
            },
            "targeted_therapy": {
                "source": "NCCN Guidelines for Non-Small Cell Lung Cancer",
                "section": "Targeted Therapy",
                "recommendation": "Targeted therapy selection based on actionable mutations (EGFR, ALK, ROS1, BRAF, MET, RET, NTRK, KRAS G12C)",
                "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450"
            }
        },
        "default": {
            "positive": {
                "source": "NCCN Guidelines",
                "section": "Treatment Recommendations",
                "recommendation": "Consider treatment options based on pathological staging and molecular profile",
                "url": "https://www.nccn.org/guidelines"
            },
            "negative": {
                "source": "NCCN Guidelines",
                "section": "Treatment Recommendations",
                "recommendation": "Review staging and consider alternative treatment approaches",
                "url": "https://www.nccn.org/guidelines"
            },
            "molecular_testing": {
                "source": "NCCN Guidelines",
                "section": "Biomarker Testing",
                "recommendation": "Molecular profiling may identify additional treatment options",
                "url": "https://www.nccn.org/guidelines"
            },
            "targeted_therapy": {
                "source": "NCCN Guidelines",
                "section": "Targeted Therapy",
                "recommendation": "Consider targeted therapy if actionable mutations are identified",
                "url": "https://www.nccn.org/guidelines"
            }
        }
    }
    
    # Legacy mapping for backwards compatibility
    NCCN_GUIDELINES = NCCN_GUIDELINES_BY_CANCER["ovarian_cancer"]
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MODERATE_CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self):
        pass
    
    def _calculate_confidence(self, score: float) -> tuple[ConfidenceLevel, float]:
        """Calculate confidence level from prediction score."""
        # Confidence is how far from 0.5 (uncertain) the score is
        confidence_score = abs(score - 0.5) * 2  # Scale to 0-1
        
        if confidence_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH, confidence_score
        elif confidence_score >= self.MODERATE_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MODERATE, confidence_score
        else:
            return ConfidenceLevel.LOW, confidence_score
    
    def _aggregate_similar_outcomes(
        self, 
        similar_cases: List[Dict[str, Any]]
    ) -> SimilarCaseOutcomes:
        """Aggregate outcomes from similar cases."""
        outcomes = SimilarCaseOutcomes()
        outcomes.total_similar = len(similar_cases)
        
        if not similar_cases:
            return outcomes
        
        similarities = []
        for case in similar_cases:
            # Check similarity score
            sim_score = case.get("similarity_score", 0)
            if sim_score > 0:
                similarities.append(sim_score)
            
            # Check outcome label if available
            label = case.get("label", "").lower()
            if "responder" in label and "non" not in label:
                outcomes.responders += 1
            elif "non-responder" in label or "non_responder" in label:
                outcomes.non_responders += 1
            else:
                outcomes.unknown += 1
        
        if similarities:
            outcomes.avg_similarity = sum(similarities) / len(similarities)
        
        return outcomes
    
    def _determine_risk_level(
        self,
        prediction: str,
        confidence_level: ConfidenceLevel,
        confidence_score: float,
        quality: QualityFactors,
        similar_outcomes: SimilarCaseOutcomes
    ) -> RiskLevel:
        """Determine overall risk stratification level."""
        
        # Quality issues reduce confidence
        if not quality.is_acceptable:
            return RiskLevel.INCONCLUSIVE
        
        # Check if similar cases align with prediction
        similar_align = True
        if similar_outcomes.has_sufficient_evidence:
            pred_is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
            similar_favor_responder = similar_outcomes.responder_ratio is not None and similar_outcomes.responder_ratio > 0.5
            similar_align = pred_is_responder == similar_favor_responder
        
        # Determine risk level
        if confidence_level == ConfidenceLevel.HIGH and similar_align:
            return RiskLevel.HIGH_CONFIDENCE
        elif confidence_level == ConfidenceLevel.MODERATE or (confidence_level == ConfidenceLevel.HIGH and not similar_align):
            return RiskLevel.MODERATE_CONFIDENCE
        elif confidence_level == ConfidenceLevel.LOW:
            if confidence_score < 0.52 - 0.5:  # Very close to 50/50
                return RiskLevel.INCONCLUSIVE
            return RiskLevel.LOW_CONFIDENCE
        
        return RiskLevel.MODERATE_CONFIDENCE
    
    def _generate_primary_recommendation(
        self,
        prediction: str,
        risk_level: RiskLevel,
        confidence_score: float,
        similar_outcomes: SimilarCaseOutcomes,
        cancer_type: str = "default",
    ) -> tuple[str, List[str]]:
        """Generate primary recommendation and supporting rationale."""
        
        is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
        # Also check for positive class indicators
        is_positive = is_responder or any(x in prediction.lower() for x in ["positive", "sensitive", "favorable", "high"])
        
        rationale = []
        
        if risk_level == RiskLevel.HIGH_CONFIDENCE:
            if is_positive:
                recommendation = (
                    "Morphological patterns suggest favorable treatment response. "
                    "Consider in context of complete clinical evaluation."
                )
                rationale = [
                    f"Model confidence is high ({confidence_score:.0%})",
                    "Tissue morphology consistent with treatment-responsive patterns",
                ]
                if similar_outcomes.has_sufficient_evidence:
                    rationale.append(
                        f"{similar_outcomes.responders} of {similar_outcomes.responders + similar_outcomes.non_responders} "
                        f"similar cases showed favorable response"
                    )
            else:
                recommendation = (
                    "Morphological patterns suggest potential treatment resistance. "
                    "Consider alternative regimens or molecular profiling for targeted options."
                )
                rationale = [
                    f"Model confidence is high ({confidence_score:.0%})",
                    "Tissue patterns associated with treatment-resistant phenotype",
                ]
                if similar_outcomes.has_sufficient_evidence:
                    rationale.append(
                        f"{similar_outcomes.non_responders} of {similar_outcomes.responders + similar_outcomes.non_responders} "
                        f"similar cases showed non-response"
                    )
        
        elif risk_level == RiskLevel.MODERATE_CONFIDENCE:
            if is_responder:
                recommendation = (
                    "Findings lean toward favorable response, but additional workup recommended "
                    "to confirm treatment strategy."
                )
            else:
                recommendation = (
                    "Findings suggest possible resistance, but recommend additional molecular "
                    "and clinical correlation before treatment decisions."
                )
            rationale = [
                f"Model confidence is moderate ({confidence_score:.0%})",
                "Additional evidence would strengthen interpretation",
            ]
        
        elif risk_level == RiskLevel.LOW_CONFIDENCE:
            recommendation = (
                f"Model confidence is low ({confidence_score:.0%}) - interpret with caution. "
                "Recommend molecular profiling and multidisciplinary review for treatment planning."
            )
            rationale = [
                "Prediction uncertainty is high",
                "Morphological patterns are not clearly indicative",
                "Clinical and molecular correlation essential",
            ]
        
        else:  # INCONCLUSIVE
            recommendation = (
                "Analysis is inconclusive. Quality issues or borderline morphology prevent "
                "reliable prediction. Recommend additional workup before treatment decisions."
            )
            rationale = [
                "Prediction could not be made with sufficient confidence",
                "Morphological or quality factors limit interpretation",
            ]
        
        return recommendation, rationale
    
    def _generate_alternative_considerations(
        self,
        prediction: str,
        risk_level: RiskLevel,
        quality: QualityFactors,
        cancer_type: str = "default",
    ) -> List[str]:
        """Generate alternative considerations for clinical discussion."""
        alternatives = []
        
        is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
        is_positive = is_responder or any(x in prediction.lower() for x in ["positive", "sensitive", "favorable"])
        
        if risk_level in [RiskLevel.MODERATE_CONFIDENCE, RiskLevel.LOW_CONFIDENCE]:
            alternatives.append(
                "Consider clinical trial enrollment if eligible"
            )
        
        if not is_positive or risk_level != RiskLevel.HIGH_CONFIDENCE:
            alternatives.append(
                "Molecular profiling may identify targeted therapy options"
            )
        
        if is_positive and risk_level == RiskLevel.HIGH_CONFIDENCE:
            alternatives.append(
                "Monitor closely for early signs of resistance during treatment"
            )
        
        if quality.quality_concerns:
            alternatives.append(
                "Consider re-evaluation with higher quality tissue section if available"
            )
        
        if not is_positive:
            alternatives.append(
                "Per NCCN guidelines, treatment-resistant disease may benefit from alternative agents or targeted therapy"
            )
        
        return alternatives
    
    def _get_guideline_references(
        self,
        prediction: str,
        risk_level: RiskLevel,
        cancer_type: str = "default"
    ) -> List[Dict[str, str]]:
        """Get relevant NCCN guideline references based on cancer type."""
        references = []
        
        # Normalize cancer type for lookup
        cancer_key = cancer_type.lower().replace(" ", "_").replace("-", "_")
        if "ovarian" in cancer_key:
            cancer_key = "ovarian_cancer"
        elif "lung" in cancer_key:
            cancer_key = "lung_cancer"
        else:
            cancer_key = "default"
        
        guidelines = self.NCCN_GUIDELINES_BY_CANCER.get(cancer_key, self.NCCN_GUIDELINES_BY_CANCER["default"])
        
        is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
        # Also check for positive class indicators like "advanced", "sensitive", etc.
        is_positive = is_responder or any(x in prediction.lower() for x in ["positive", "sensitive", "advanced", "high"])
        
        if is_positive:
            references.append(guidelines["positive"])
        else:
            references.append(guidelines["negative"])
        
        # Always recommend molecular testing
        references.append(guidelines["molecular_testing"])
        
        # Add targeted therapy reference
        references.append(guidelines["targeted_therapy"])
        
        return references
    
    def _generate_uncertainty_statement(
        self,
        confidence_level: ConfidenceLevel,
        confidence_score: float,
        quality: QualityFactors
    ) -> str:
        """Generate clear uncertainty messaging."""
        
        if confidence_level == ConfidenceLevel.HIGH and quality.is_acceptable:
            return (
                f"Model confidence is high ({confidence_score:.0%}). While the prediction "
                f"is well-supported by morphological patterns, clinical correlation remains essential."
            )
        elif confidence_level == ConfidenceLevel.MODERATE:
            return (
                f"Model confidence is moderate ({confidence_score:.0%}). The prediction "
                f"should be interpreted with caution and validated through additional clinical "
                f"and molecular assessment."
            )
        elif confidence_level == ConfidenceLevel.LOW:
            return (
                f"Model confidence is low ({confidence_score:.0%}). This prediction has "
                f"significant uncertainty and should not be used as the primary basis for "
                f"treatment decisions without extensive additional workup."
            )
        else:
            if not quality.is_acceptable:
                return (
                    "Slide quality issues limit the reliability of this analysis. "
                    "Results should be interpreted with significant caution."
                )
            return (
                "Prediction confidence cannot be reliably assessed. "
                "Recommend comprehensive clinical evaluation."
            )
    
    def _generate_suggested_workup(
        self,
        prediction: str,
        risk_level: RiskLevel,
        quality: QualityFactors,
        cancer_type: str = "default",
    ) -> List[str]:
        """Generate suggested additional workup."""
        workup = []
        
        is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
        is_positive = is_responder or any(x in prediction.lower() for x in ["positive", "sensitive", "favorable"])
        
        # Normalize cancer type for specific recommendations
        cancer_key = cancer_type.lower().replace(" ", "_").replace("-", "_")
        is_ovarian = "ovarian" in cancer_key
        is_lung = "lung" in cancer_key
        
        # Always recommend certain baseline evaluations
        if risk_level != RiskLevel.HIGH_CONFIDENCE:
            if is_ovarian:
                workup.append("Molecular profiling for BRCA1/2 mutations and HRD status")
            elif is_lung:
                workup.append("Broad molecular profiling (EGFR, ALK, ROS1, BRAF, KRAS, PD-L1)")
            else:
                workup.append("Molecular profiling to identify actionable mutations")
        
        if not is_positive:
            workup.append("Consider tissue-based biomarker testing for targeted therapy eligibility")
            workup.append("Review treatment history and prior therapy intervals")
        
        if risk_level in [RiskLevel.LOW_CONFIDENCE, RiskLevel.INCONCLUSIVE]:
            workup.append("Multidisciplinary tumor board review recommended")
            workup.append("Consider additional pathology review of key tissue regions")
        
        if quality.quality_concerns:
            workup.append("Re-section and re-analyze if higher quality tissue available")
        
        # Cancer-type-specific tumor markers
        if is_ovarian:
            workup.append("Correlate with imaging findings and tumor markers (CA-125)")
        elif is_lung:
            workup.append("Correlate with imaging findings and relevant biomarkers")
        else:
            workup.append("Correlate with imaging findings and relevant tumor markers")
        
        return workup
    
    def _generate_interpretation_note(
        self,
        prediction: str,
        confidence_score: float,
        similar_outcomes: SimilarCaseOutcomes
    ) -> str:
        """Generate interpretation guidance note."""
        
        is_responder = "responder" in prediction.lower() and "non" not in prediction.lower()
        
        note = f"The model predicts {'favorable response' if is_responder else 'potential resistance'} "
        note += f"with {confidence_score:.0%} confidence. "
        
        if similar_outcomes.has_sufficient_evidence:
            ratio = similar_outcomes.responder_ratio
            if ratio is not None:
                note += f"In {similar_outcomes.responders + similar_outcomes.non_responders} morphologically similar cases, "
                note += f"{ratio:.0%} showed favorable treatment response. "
        
        note += (
            "This morphological assessment should be integrated with clinical staging, "
            "molecular profiling, and patient factors for comprehensive treatment planning."
        )
        
        return note
    
    def generate(
        self,
        prediction: str,
        score: float,
        similar_cases: List[Dict[str, Any]],
        quality_metrics: Optional[Dict[str, Any]] = None,
        patient_context: Optional[Dict[str, Any]] = None,
        cancer_type: str = "default"
    ) -> DecisionSupportOutput:
        """
        Generate comprehensive clinical decision support.
        
        Args:
            prediction: Model prediction label ("RESPONDER" or "NON-RESPONDER")
            score: Model prediction score (0-1)
            similar_cases: List of similar case dictionaries
            quality_metrics: Optional slide quality metrics
            patient_context: Optional patient clinical context
            cancer_type: Cancer type for guideline lookup (e.g., "Ovarian Cancer", "Lung Cancer")
            
        Returns:
            DecisionSupportOutput with structured recommendations
        """
        # Calculate confidence
        confidence_level, confidence_score = self._calculate_confidence(score)
        
        # Process quality factors
        quality = QualityFactors()
        if quality_metrics:
            quality.slide_quality = quality_metrics.get("overall_quality", "unknown")
            quality.tissue_coverage = quality_metrics.get("tissue_coverage")
            quality.blur_score = quality_metrics.get("blur_score")
            quality.artifact_detected = quality_metrics.get("artifact_detected", False)
        
        # Aggregate similar case outcomes
        similar_outcomes = self._aggregate_similar_outcomes(similar_cases)
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            prediction, confidence_level, confidence_score, quality, similar_outcomes
        )
        
        # Generate recommendations
        primary_rec, rationale = self._generate_primary_recommendation(
            prediction, risk_level, confidence_score, similar_outcomes, cancer_type
        )
        
        alternatives = self._generate_alternative_considerations(
            prediction, risk_level, quality, cancer_type
        )
        
        guideline_refs = self._get_guideline_references(prediction, risk_level, cancer_type)
        
        uncertainty = self._generate_uncertainty_statement(
            confidence_level, confidence_score, quality
        )
        
        workup = self._generate_suggested_workup(prediction, risk_level, quality, cancer_type)
        
        interpretation = self._generate_interpretation_note(
            prediction, confidence_score, similar_outcomes
        )
        
        return DecisionSupportOutput(
            risk_level=risk_level,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            primary_recommendation=primary_rec,
            supporting_rationale=rationale,
            alternative_considerations=alternatives,
            guideline_references=guideline_refs,
            uncertainty_statement=uncertainty,
            quality_warnings=quality.quality_concerns,
            suggested_workup=workup,
            interpretation_note=interpretation
        )
    
    def to_dict(self, output: DecisionSupportOutput) -> Dict[str, Any]:
        """Convert DecisionSupportOutput to dictionary for JSON serialization."""
        return {
            "risk_level": output.risk_level.value,
            "confidence_level": output.confidence_level.value,
            "confidence_score": output.confidence_score,
            "primary_recommendation": output.primary_recommendation,
            "supporting_rationale": output.supporting_rationale,
            "alternative_considerations": output.alternative_considerations,
            "guideline_references": output.guideline_references,
            "uncertainty_statement": output.uncertainty_statement,
            "quality_warnings": output.quality_warnings,
            "suggested_workup": output.suggested_workup,
            "interpretation_note": output.interpretation_note,
            "caveat": output.caveat
        }
