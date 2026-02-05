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
import threading
import time
import inspect

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportingConfig:
    """MedGemma reporting configuration."""
    # Use local path in container, fallback to HuggingFace ID
    model: str = "/app/models/medgemma-4b-it"
    max_evidence_patches: int = 8
    max_similar_cases: int = 5
    max_input_tokens: int = 3072
    max_output_tokens: int = 512
    max_generation_time_s: float = 30.0
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
        self._load_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._warmup_done = False
        self._effective_max_input_tokens = None
        self._supports_max_time = None

    def _load_model(self) -> None:
        """Load MedGemma model."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

            # Determine device
            # Force CPU to avoid CUDA driver issues on Blackwell GPUs
            use_cpu = True  # Set to False to re-enable CUDA when driver is updated
            if not use_cpu and torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            logger.info(f"Loading MedGemma model on {self._device}")

            # Load tokenizer and model
            model_id = self.config.model

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Try to load processor for multimodal input
            try:
                self._processor = AutoProcessor.from_pretrained(model_id)
            except Exception:
                logger.warning("Could not load processor; using text-only mode")
                self._processor = None

            # Load model with appropriate precision
            model_kwargs = {
                "torch_dtype": torch.float16 if self._device.type == "cuda" else torch.float32,
                "device_map": "auto" if self._device.type == "cuda" else None,
                "low_cpu_mem_usage": True,
            }
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs,
                )
            except TypeError:
                model_kwargs.pop("low_cpu_mem_usage", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs,
                )

            if self._device.type != "cuda":
                self._model = self._model.to(self._device)

            self._model.eval()

            # Track effective max input tokens for prompt trimming
            model_max = getattr(self._tokenizer, "model_max_length", None)
            if model_max is None or model_max > 100000:
                model_max = self.config.max_input_tokens
            self._effective_max_input_tokens = min(self.config.max_input_tokens, model_max)

            # Ensure generation config has sane padding defaults
            if getattr(self._model, "generation_config", None) is not None:
                if self._model.generation_config.pad_token_id is None:
                    self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id
                if self._model.generation_config.eos_token_id is None:
                    self._model.generation_config.eos_token_id = self._tokenizer.eos_token_id

            try:
                self._supports_max_time = "max_time" in inspect.signature(self._model.generate).parameters
            except (TypeError, ValueError):
                self._supports_max_time = False

            logger.info("MedGemma model loaded successfully")


    def _warmup_inference(self) -> None:
        """Run a test inference to warm up CUDA kernels."""
        if self._warmup_done:
            return

        with self._warmup_lock:
            if self._warmup_done:
                return

            import torch

            self._load_model()

            # Short test prompt
            test_prompt = "Describe healthy tissue."

            try:
                inputs = self._tokenizer(
                    test_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.inference_mode():
                    self._model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                logger.info("MedGemma warmup inference completed")
            except Exception as e:
                logger.warning(f"MedGemma warmup inference failed: {e}")
            finally:
                self._warmup_done = True

    def _format_patient_context(self, patient_context: Optional[Dict]) -> str:
        """Format patient context into a clinical description."""
        if not patient_context:
            return "No patient demographics available."

        parts = []
        if patient_context.get("age"):
            parts.append(f"{patient_context['age']}-year-old")
        if patient_context.get("sex"):
            sex_full = "female" if patient_context["sex"].upper() == "F" else "male" if patient_context["sex"].upper() == "M" else patient_context["sex"]
            parts.append(sex_full)

        desc = " ".join(parts) if parts else "Patient"

        clinical_parts = []
        if patient_context.get("stage"):
            clinical_parts.append(f"Stage {patient_context['stage']}")
        if patient_context.get("grade"):
            clinical_parts.append(f"{patient_context['grade']} grade")
        if patient_context.get("histology"):
            clinical_parts.append(patient_context["histology"])
        if clinical_parts:
            desc += " with " + ", ".join(clinical_parts)

        if patient_context.get("prior_lines") is not None:
            lines = patient_context["prior_lines"]
            if lines == 0:
                desc += ". Treatment-naive."
            else:
                desc += f". {lines} prior line{'s' if lines > 1 else ''} of therapy."

        return desc

    def _build_prompt(
        self,
        evidence_patches: List[Dict],
        score: float,
        label: str,
        similar_cases: List[Dict],
        case_id: str = "unknown",
        patient_context: Optional[Dict] = None,
    ) -> str:
        """Build the prompt for MedGemma."""

        # Format patient context
        patient_desc = self._format_patient_context(patient_context)

        # Build evidence description
        evidence_text = "\n".join([
            f"- Patch {p['rank']}: Attention weight {p['attention_weight']:.3f}, "
            f"coordinates ({p['coordinates'][0]}, {p['coordinates'][1]})"
            for p in evidence_patches[:self.config.max_evidence_patches]
        ])

        # Build similar cases description
        similar_text = ""
        if similar_cases and self.config.max_similar_cases > 0:
            similar_cases_deduped = []
            seen = set()
            for s in similar_cases[:self.config.max_similar_cases]:
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

PATIENT CONTEXT:
{patient_desc}

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

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for the prompt."""
        if self._tokenizer is None:
            return max(1, len(prompt.split()))
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

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
        required_fields = [
            "case_id", "task", "model_output", "evidence",
            "limitations", "safety_statement"
        ]

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
        patient_context: Optional[Dict] = None,
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
            patient_context: Patient demographic/clinical context

        Returns:
            Structured report dictionary
        """
        self._load_model()
        if not self._warmup_done:
            self._warmup_inference()

        import torch

        evidence_limit = min(len(evidence_patches), self.config.max_evidence_patches)
        similar_limit = min(len(similar_cases), self.config.max_similar_cases)

        prompt = self._build_prompt(
            evidence_patches[:evidence_limit],
            score,
            label,
            similar_cases[:similar_limit],
            case_id,
            patient_context,
        )
        prompt_tokens = self._count_prompt_tokens(prompt)

        if self._effective_max_input_tokens is None:
            self._effective_max_input_tokens = self.config.max_input_tokens

        if prompt_tokens > self._effective_max_input_tokens:
            logger.warning(
                "Prompt too long (%d tokens > %d); dropping similar cases",
                prompt_tokens,
                self._effective_max_input_tokens,
            )
            if similar_limit > 0:
                similar_limit = 0
                prompt = self._build_prompt(
                    evidence_patches[:evidence_limit],
                    score,
                    label,
                    [],
                    case_id,
                    patient_context,
                )
                prompt_tokens = self._count_prompt_tokens(prompt)

        while prompt_tokens > self._effective_max_input_tokens and evidence_limit > 3:
            evidence_limit -= 1
            prompt = self._build_prompt(
                evidence_patches[:evidence_limit],
                score,
                label,
                [],
                case_id,
                patient_context,
            )
            prompt_tokens = self._count_prompt_tokens(prompt)

        if prompt_tokens > self._effective_max_input_tokens:
            logger.warning(
                "Prompt still long (%d tokens); tokenizer will truncate to %d tokens",
                prompt_tokens,
                self._effective_max_input_tokens,
            )

        for attempt in range(max_retries + 1):
            try:
                with self._generate_lock:
                    # Tokenize
                    inputs = self._tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=prompt_tokens > self._effective_max_input_tokens,
                        max_length=self._effective_max_input_tokens,
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    max_new_tokens = max(64, int(self.config.max_output_tokens))
                    max_time = self.config.max_generation_time_s

                    logger.info(
                        "MedGemma generation start (prompt_tokens=%d, max_new_tokens=%d, max_time=%.1fs)",
                        prompt_tokens,
                        max_new_tokens,
                        max_time,
                    )
                    start_time = time.time()

                    # Generate
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "do_sample": self.config.temperature > 0,
                        "pad_token_id": self._tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    if self._supports_max_time:
                        gen_kwargs["max_time"] = max_time

                    with torch.inference_mode():
                        outputs = self._model.generate(
                            **inputs,
                            **gen_kwargs,
                        )

                    gen_elapsed = time.time() - start_time
                    hit_time_limit = (
                        bool(self._supports_max_time)
                        and max_time is not None
                        and gen_elapsed >= max_time - 0.5
                    )
                    logger.info(
                        "MedGemma generation finished in %.1fs",
                        gen_elapsed,
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
                    if hit_time_limit:
                        logger.warning("MedGemma hit generation time limit; skipping retries")
                        break

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

    def generate_report(
        self,
        evidence_patches: List[Dict],
        score: float,
        label: str,
        similar_cases: List[Dict],
        case_id: str = "unknown",
        patient_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a complete report with structured data and summary.

        This is a convenience method that calls generate() and generate_summary()
        together and returns both in a single response.

        Args:
            evidence_patches: List of evidence patch dicts
            score: Model prediction score
            label: Predicted label
            similar_cases: List of similar case dicts
            case_id: Case identifier
            patient_context: Patient demographic/clinical context

        Returns:
            Dict with 'structured' and 'summary' keys
        """
        # Generate structured report
        structured = self.generate(
            evidence_patches=evidence_patches,
            score=score,
            label=label,
            similar_cases=similar_cases,
            case_id=case_id,
            patient_context=patient_context,
        )

        # Generate human-readable summary
        summary = self.generate_summary(structured)

        return {
            "structured": structured,
            "summary": summary,
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
