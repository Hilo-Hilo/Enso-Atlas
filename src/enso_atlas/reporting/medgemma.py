"""
MedGemma Reporter - Structured report generation from evidence.

Uses MedGemma 1.5 to generate:
- Structured JSON reports for auditing
- Human-readable tumor board summaries
- Safety-aware, grounded interpretations
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import json
import re

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportingConfig:
    """MedGemma reporting configuration."""
    model: str = "google/medgemma-4b-it"
    max_evidence_patches: int = 8
    max_output_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9


# Report schema for validation
REPORT_SCHEMA = {
    "type": "object",
    "required": ["case_id", "task", "model_output", "evidence", "limitations", "safety_statement"],
    "properties": {
        "case_id": {"type": "string"},
        "task": {"type": "string"},
        "model_output": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "probability": {"type": "number"},
                "calibration_note": {"type": "string"}
            }
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "patch_id": {"type": "string"},
                    "morphology_description": {"type": "string"},
                    "significance": {"type": "string"}
                }
            }
        },
        "similar_examples": {"type": "array"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "suggested_next_steps": {"type": "array", "items": {"type": "string"}},
        "safety_statement": {"type": "string"}
    }
}


class MedGemmaReporter:
    """
    Report generator using MedGemma.
    
    Produces structured, safety-aware reports grounded in the
    evidence patches and model predictions. Designed for tumor
    board discussion and clinical documentation.
    """
    
    def __init__(self, config: ReportingConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None
    
    def _load_model(self) -> None:
        """Load MedGemma model."""
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Loading MedGemma model on {self._device}")
        
        # Load tokenizer and model
        model_id = self.config.model
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Try to load processor for multimodal input
        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            logger.warning("Could not load processor; using text-only mode")
            self._processor = None
        
        # Load model with appropriate precision
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
            device_map="auto" if self._device.type == "cuda" else None,
        )
        
        if self._device.type != "cuda":
            self._model = self._model.to(self._device)
        
        self._model.eval()
        logger.info("MedGemma model loaded successfully")
    
    def _build_prompt(
        self,
        evidence_patches: List[Dict],
        score: float,
        label: str,
        similar_cases: List[Dict],
        case_id: str = "unknown",
    ) -> str:
        """Build the prompt for MedGemma."""
        
        # Build evidence description
        evidence_text = "\n".join([
            f"- Patch {p['rank']}: Attention weight {p['attention_weight']:.3f}, "
            f"coordinates ({p['coordinates'][0]}, {p['coordinates'][1]})"
            for p in evidence_patches[:self.config.max_evidence_patches]
        ])
        
        # Build similar cases description
        similar_text = ""
        if similar_cases:
            similar_cases_deduped = []
            seen = set()
            for s in similar_cases[:5]:
                case_key = s.get("metadata", {}).get("slide_id", "unknown")
                if case_key not in seen:
                    seen.add(case_key)
                    similar_cases_deduped.append(s)
            
            similar_text = "\n".join([
                f"- Similar case: {s.get('metadata', {}).get('slide_id', 'unknown')}, "
                f"distance: {s['distance']:.3f}"
                for s in similar_cases_deduped
            ])
        
        prompt = f"""You are a medical AI assistant helping prepare a tumor board summary for a pathology case. You must be cautious, factual, and clearly state limitations.

CASE ID: {case_id}

MODEL PREDICTION:
- Classification: {label}
- Probability score: {score:.3f}
- Task: Bevacizumab treatment response prediction

EVIDENCE PATCHES (ordered by model attention):
{evidence_text}

SIMILAR CASES FROM REFERENCE COHORT:
{similar_text if similar_text else "No similar cases available in reference cohort."}

Generate a structured JSON report with the following format:
{{
    "case_id": "{case_id}",
    "task": "Bevacizumab treatment response prediction from H&E histopathology",
    "model_output": {{
        "label": "{label}",
        "probability": {score:.3f},
        "calibration_note": "Model probability, not clinical certainty. Requires external validation."
    }},
    "evidence": [
        {{
            "patch_id": "patch_N",
            "morphology_description": "Brief description of visible morphology",
            "significance": "Why this region may be relevant"
        }}
    ],
    "similar_examples": [],
    "limitations": [
        "List specific limitations of this analysis"
    ],
    "suggested_next_steps": [
        "Suggested confirmatory tests or additional evaluation"
    ],
    "safety_statement": "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists. Do not use for standalone clinical decision-making."
}}

IMPORTANT CONSTRAINTS:
1. Do NOT recommend specific treatments or drugs
2. Do NOT claim clinical certainty
3. DO cite specific patch IDs when describing evidence
4. DO acknowledge model limitations
5. Keep descriptions factual and based only on the provided evidence

Generate the JSON report:"""
        
        return prompt
    
    def _parse_json_response(self, response: str) -> Dict:
        """Extract and parse JSON from model response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # Return a minimal valid structure if parsing fails
        return {
            "case_id": "unknown",
            "task": "Report generation failed",
            "model_output": {
                "label": "unknown",
                "probability": 0.0,
                "calibration_note": "Report generation encountered an error"
            },
            "evidence": [],
            "limitations": ["Report generation failed - manual review required"],
            "suggested_next_steps": ["Manual pathology review"],
            "safety_statement": "This report could not be generated properly. Do not use for any clinical purpose.",
            "raw_response": response
        }
    
    def _validate_report(self, report: Dict) -> bool:
        """Validate report against schema."""
        required_fields = ["case_id", "task", "model_output", "evidence", 
                          "limitations", "safety_statement"]
        
        for field in required_fields:
            if field not in report:
                logger.warning(f"Report missing required field: {field}")
                return False
        
        # Check for prohibited statements
        prohibited_phrases = [
            "start treatment",
            "stop treatment",
            "recommend starting",
            "recommend stopping",
            "prescribe",
            "administer",
        ]
        
        report_str = json.dumps(report).lower()
        for phrase in prohibited_phrases:
            if phrase in report_str:
                logger.warning(f"Report contains prohibited phrase: {phrase}")
                return False
        
        return True
    
    def generate(
        self,
        evidence_patches: List[Dict],
        score: float,
        label: str,
        similar_cases: List[Dict],
        case_id: str = "unknown",
        max_retries: int = 2,
    ) -> Dict:
        """
        Generate a structured report.
        
        Args:
            evidence_patches: List of evidence patch dicts
            score: Model prediction score
            label: Predicted label
            similar_cases: List of similar case dicts
            case_id: Case identifier
            max_retries: Number of retries on failure
            
        Returns:
            Structured report dictionary
        """
        self._load_model()
        
        import torch
        
        prompt = self._build_prompt(
            evidence_patches, score, label, similar_cases, case_id
        )
        
        for attempt in range(max_retries + 1):
            try:
                # Tokenize
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_output_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                
                # Decode
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Parse JSON
                report = self._parse_json_response(response)
                
                # Validate
                if self._validate_report(report):
                    logger.info("Generated valid report")
                    return report
                else:
                    logger.warning(f"Report validation failed, attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Report generation error (attempt {attempt + 1}): {e}")
        
        # Return fallback report
        return self._create_fallback_report(case_id, score, label)
    
    def _create_fallback_report(
        self,
        case_id: str,
        score: float,
        label: str,
    ) -> Dict:
        """Create a fallback report when generation fails."""
        return {
            "case_id": case_id,
            "task": "Bevacizumab treatment response prediction from H&E histopathology",
            "model_output": {
                "label": label,
                "probability": score,
                "calibration_note": "Automated report generation failed. Manual interpretation required."
            },
            "evidence": [],
            "limitations": [
                "Automated report generation failed",
                "Evidence interpretation not available",
                "Manual pathology review required"
            ],
            "suggested_next_steps": [
                "Manual review of evidence patches",
                "Consultation with pathology team"
            ],
            "safety_statement": "This is a research tool. Report generation encountered errors. All findings require manual validation by qualified pathologists."
        }
    
    def generate_summary(self, report: Dict) -> str:
        """
        Generate a human-readable summary from structured report.
        
        Args:
            report: Structured report dictionary
            
        Returns:
            Formatted text summary
        """
        model_output = report.get("model_output", {})
        evidence = report.get("evidence", [])
        limitations = report.get("limitations", [])
        next_steps = report.get("suggested_next_steps", [])
        safety = report.get("safety_statement", "")
        
        summary = f"""
=== TUMOR BOARD SUMMARY ===

Case ID: {report.get('case_id', 'Unknown')}
Task: {report.get('task', 'Unknown')}

PREDICTION:
- Classification: {model_output.get('label', 'Unknown')}
- Score: {model_output.get('probability', 0):.3f}
- Note: {model_output.get('calibration_note', '')}

EVIDENCE PATCHES:
"""
        for e in evidence[:5]:
            summary += f"  - {e.get('patch_id', 'Unknown')}: {e.get('morphology_description', '')}\n"
            summary += f"    Significance: {e.get('significance', '')}\n"
        
        summary += f"""
LIMITATIONS:
"""
        for lim in limitations:
            summary += f"  - {lim}\n"
        
        summary += f"""
SUGGESTED NEXT STEPS:
"""
        for step in next_steps:
            summary += f"  - {step}\n"
        
        summary += f"""
SAFETY STATEMENT:
{safety}

=== END OF SUMMARY ===
"""
        return summary
