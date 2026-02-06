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


def _patch_torch_autocast():
    """Patch torch.is_autocast_enabled for transformers 5.0+ compatibility.
    
    Newer transformers calls torch.is_autocast_enabled() with no arguments,
    but some PyTorch versions have a different signature. This patches it to work.
    """
    import torch
    
    original_is_autocast_enabled = torch.is_autocast_enabled
    
    def patched_is_autocast_enabled(*args, **kwargs):
        try:
            return original_is_autocast_enabled()
        except TypeError:
            # Old signature expected an argument
            if args or kwargs:
                return original_is_autocast_enabled(*args, **kwargs)
            # Try to get CPU autocast status
            try:
                return torch.is_autocast_cpu_enabled() or torch.is_autocast_cuda_enabled()
            except Exception:
                return False
    
    torch.is_autocast_enabled = patched_is_autocast_enabled
    logger.info("Patched torch.is_autocast_enabled for transformers compatibility")


# Apply patch on module load
try:
    _patch_torch_autocast()
except Exception as e:
    logger.warning(f"Failed to patch torch.is_autocast_enabled: {e}")


@dataclass
class ReportingConfig:
    """MedGemma reporting configuration."""
    # Use local path in container, fallback to HuggingFace ID
    model: str = "/app/models/medgemma-4b-it"
    max_evidence_patches: int = 4  # Reduced for shorter prompts
    max_similar_cases: int = 0  # Skip similar cases to reduce prompt size
    max_input_tokens: int = 512  # Simplified prompt is much shorter
    max_output_tokens: int = 256  # Small enough to finish within CPU time limit
    max_generation_time_s: float = 90.0  # Leave headroom for decode + parse after generation
    temperature: float = 0.1  # Lower temp for more predictable JSON
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
        self._supports_stopping_criteria = None

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
            # Force CPU to avoid CUDA compatibility issues on Blackwell GPUs (sm_121)
            # PyTorch in the container doesn't support sm_121 arch, causing hangs.
            # CPU inference is reliable: 4B model in bfloat16 uses ~8GB of 128GB RAM.
            use_cpu = True  # Set to False to re-enable CUDA when PyTorch adds sm_121 support
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
            # Use bfloat16 on CPU (supported on modern ARM CPUs) for ~8GB memory usage
            # Fall back to float32 if bfloat16 fails (e.g., older ARM without BF16 support)
            if self._device.type == "cuda":
                dtype = torch.bfloat16
                device_map = "auto"
            else:
                # Try bfloat16 on CPU first (works on modern ARM and x86 with AVX-512)
                try:
                    _test = torch.tensor([1.0], dtype=torch.bfloat16)
                    dtype = torch.bfloat16
                    logger.info("CPU supports bfloat16, using bfloat16 for model (~8GB)")
                except Exception:
                    dtype = torch.float32
                    logger.info("CPU does not support bfloat16, falling back to float32 (~16GB)")
                device_map = None

            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "low_cpu_mem_usage": True,
            }
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs,
                )
            except (TypeError, RuntimeError) as load_err:
                logger.warning(f"Model load with bfloat16 failed ({load_err}), retrying with float32")
                model_kwargs["torch_dtype"] = torch.float32
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
                generate_params = inspect.signature(self._model.generate).parameters
                self._supports_max_time = "max_time" in generate_params
                self._supports_stopping_criteria = "stopping_criteria" in generate_params
            except (TypeError, ValueError):
                self._supports_max_time = False
                self._supports_stopping_criteria = False

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

                t0 = time.time()
                with torch.inference_mode():
                    self._model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                duration = time.time() - t0

                logger.info(f"MedGemma warmup inference completed in {duration:.1f}s")
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

    # Tissue categories mapped from patch index (matches API classify_tissue_type)
    TISSUE_TYPES = ["tumor", "stroma", "necrosis", "inflammatory", "normal", "artifact"]

    def _classify_patch(self, patch: Dict) -> str:
        """Derive tissue category for a patch, matching the API's classify_tissue_type logic."""
        # Use explicit category/tissue_type if present
        cat = patch.get("category") or patch.get("tissue_type")
        if cat and cat != "unknown":
            return cat
        # Fallback: deterministic from patch_index (same as API)
        idx = patch.get("patch_index", 0)
        return self.TISSUE_TYPES[idx % len(self.TISSUE_TYPES)]

    def _build_prompt(
        self,
        evidence_patches: List[Dict],
        score: float,
        label: str,
        similar_cases: List[Dict],
        case_id: str = "unknown",
        patient_context: Optional[Dict] = None,
    ) -> str:
        """Build a compact prompt that forces JSON-first output from MedGemma."""

        # Collect tissue categories from evidence patches
        categories = []
        for p in evidence_patches[:3]:
            cat = self._classify_patch(p)
            attn = p.get("attention_weight", 0.0)
            categories.append(f"{cat}({attn:.2f})")
        cat_str = ", ".join(categories) if categories else "mixed"

        prompt = f"""Ovarian H&E. Prediction: {label} ({score:.2f}). Tissue: {cat_str}.
{{"prediction":"{label}","confidence":{score:.2f},"morphology":""""

        return prompt

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for the prompt."""
        if self._tokenizer is None:
            return max(1, len(prompt.split()))
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    def _parse_json_response(self, response: str, case_id: str = "unknown", score: float = 0.0, label: str = "unknown") -> Dict:
        """Extract and parse JSON from model response, mapping simplified output to full report structure."""
        # Extract the FIRST JSON object from markdown code blocks
        # Model sometimes repeats the JSON multiple times in fences
        code_block_match = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            # Fallback: find first JSON object directly
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            json_str = json_match.group() if json_match else None
        
        # Legacy fallback for any JSON
        if not json_str:
            cleaned = response.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            json_match = re.search(r'\{[\s\S]*?\}', cleaned)
            json_str = json_match.group() if json_match else None
        
        parsed = None
        if json_str:
            # Try to fix truncated JSON by closing brackets
            try:
                parsed = json.loads(json_str)
                logger.info("Successfully parsed JSON from MedGemma response")
            except json.JSONDecodeError:
                # Try to repair truncated JSON with increasingly aggressive fixes
                repair_suffixes = [
                    '"}',        # truncated string value
                    '"]',        # truncated array
                    '"]}',       # truncated array in object
                    '"}]}',      # truncated string in array in object
                    '"]}}}',     # deeply nested
                    '..."}],"clinical_significance":"See key findings","recommendation":"Correlate with clinical context"}',
                ]
                for fix in repair_suffixes:
                    try:
                        parsed = json.loads(json_str + fix)
                        logger.info("Repaired truncated JSON with suffix: %s", fix)
                        break
                    except json.JSONDecodeError:
                        continue
                if parsed is None:
                    # Last resort: extract whatever fields we can with regex
                    logger.warning("JSON repair failed, attempting field extraction")
                    parsed = {}
                    for field in ["prediction", "morphology_description", "clinical_significance", "recommendation"]:
                        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', json_str)
                        if m:
                            parsed[field] = m.group(1)
                    # Extract key_findings array
                    kf_match = re.search(r'"key_findings"\s*:\s*\[(.*)', json_str)
                    if kf_match:
                        findings = re.findall(r'"([^"]+)"', kf_match.group(1))
                        if findings:
                            parsed["key_findings"] = findings
                    if parsed:
                        logger.info("Extracted %d fields via regex fallback", len(parsed))
                    else:
                        logger.warning("Failed to parse JSON even after repair and regex attempts")

        # Map simplified response to full report structure
        if parsed:
            key_findings = parsed.get("key_findings", [])
            recommendation = parsed.get("recommendation", "Review with pathology team")
            morphology_desc = parsed.get("morphology_description", "")
            clinical_sig = parsed.get("clinical_significance", "")
            
            # Build evidence entries from key_findings, enriched with morphology
            evidence_entries = []
            for i, f in enumerate(key_findings[:4]):
                evidence_entries.append({
                    "patch_id": f"patch_{i+1}",
                    "morphology_description": f,
                    "significance": "High attention region identified by CLAM model",
                })
            
            # If morphology_description is present, add it as a top-level evidence entry
            if morphology_desc and not any(morphology_desc in e.get("morphology_description", "") for e in evidence_entries):
                evidence_entries.insert(0, {
                    "patch_id": "overall_morphology",
                    "morphology_description": morphology_desc,
                    "significance": "Overall morphological assessment of high-attention regions",
                })
            
            return {
                "case_id": case_id,
                "task": "Bevacizumab treatment response prediction from H&E histopathology",
                "model_output": {
                    "label": parsed.get("prediction", label),
                    "probability": parsed.get("confidence", score),
                    "calibration_note": "Model probability, not clinical certainty. Requires external validation.",
                    "clinical_significance": clinical_sig,
                },
                "evidence": evidence_entries,
                "limitations": [
                    "AI prediction requires pathologist validation",
                    "Based on H&E morphology only — no IHC or molecular data",
                    "Training cohort may not represent all patient populations",
                ],
                "suggested_next_steps": [recommendation] if recommendation else ["Pathology review recommended"],
                "safety_statement": "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists.",
            }

        # Return a minimal valid structure if parsing fails
        return {
            "case_id": case_id,
            "task": "Bevacizumab treatment response prediction from H&E histopathology",
            "model_output": {
                "label": label,
                "probability": score,
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
                    logger.info(
                        "MedGemma tokenizing prompt (prompt_tokens=%d, max_input_tokens=%d)",
                        prompt_tokens,
                        self._effective_max_input_tokens,
                    )
                    inputs = self._tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=prompt_tokens > self._effective_max_input_tokens,
                        max_length=self._effective_max_input_tokens,
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    input_len = int(inputs["input_ids"].shape[1])
                    logger.info("MedGemma tokenization complete (input_ids=%d)", input_len)

                    configured_max_new_tokens = int(self.config.max_output_tokens)
                    if configured_max_new_tokens <= 0:
                        configured_max_new_tokens = 256
                    if configured_max_new_tokens > 2048:
                        logger.warning(
                            "max_output_tokens=%d too high; capping to 2048",
                            configured_max_new_tokens,
                        )
                        configured_max_new_tokens = 2048
                    max_new_tokens = max(64, configured_max_new_tokens)

                    max_time = self.config.max_generation_time_s
                    if max_time is not None:
                        try:
                            max_time = float(max_time)
                        except (TypeError, ValueError):
                            max_time = None
                    if max_time is not None and max_time <= 0:
                        max_time = None

                    max_time_display = f"{max_time:.1f}s" if max_time is not None else "none"

                    logger.info(
                        "MedGemma generation start (prompt_tokens=%d, input_ids=%d, max_new_tokens=%d, max_time=%s)",
                        prompt_tokens,
                        input_len,
                        max_new_tokens,
                        max_time_display,
                    )

                    # Generate
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "do_sample": self.config.temperature > 0,
                        "pad_token_id": self._tokenizer.eos_token_id,
                        "eos_token_id": self._tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    gen_start = time.monotonic()
                    if self._supports_max_time and max_time is not None:
                        gen_kwargs["max_time"] = max_time
                    elif max_time is not None and self._supports_stopping_criteria:
                        try:
                            from transformers.generation.stopping_criteria import (
                                StoppingCriteria,
                                StoppingCriteriaList,
                            )

                            class _TimeLimitCriteria(StoppingCriteria):
                                def __init__(self, start_time: float, max_time_s: float):
                                    self._start_time = start_time
                                    self._max_time_s = max_time_s

                                def __call__(self, input_ids, scores, **kwargs):
                                    return (time.monotonic() - self._start_time) >= self._max_time_s

                            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                                [_TimeLimitCriteria(gen_start, max_time)]
                            )
                            logger.info(
                                "MedGemma using stopping criteria for time limit (%.1fs)",
                                max_time,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to set time limit stopping criteria: %s",
                                e,
                            )
                    elif max_time is not None:
                        logger.warning(
                            "MedGemma generate() does not support max_time or stopping_criteria; "
                            "generation may run long"
                        )

                    with torch.inference_mode():
                        outputs = self._model.generate(
                            **inputs,
                            **gen_kwargs,
                        )

                    gen_elapsed = time.monotonic() - gen_start
                    output_ids = outputs[0]
                    new_tokens = int(output_ids.shape[0] - input_len)
                    hit_time_limit = (
                        max_time is not None
                        and gen_elapsed >= max_time - 0.5
                    )
                    if hit_time_limit:
                        logger.warning("MedGemma generation reached time limit (%.1fs)", max_time)
                    if new_tokens >= max_new_tokens:
                        logger.warning(
                            "MedGemma generation hit max_new_tokens=%d (new_tokens=%d)",
                            max_new_tokens,
                            new_tokens,
                        )
                    logger.info(
                        "MedGemma generation finished in %.1fs (new_tokens=%d)",
                        gen_elapsed,
                        new_tokens,
                    )

                # Decode
                decode_start = time.monotonic()
                response = self._tokenizer.decode(
                    output_ids[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                decode_elapsed = time.monotonic() - decode_start
                logger.info(
                    "MedGemma decode finished in %.2fs (chars=%d)",
                    decode_elapsed,
                    len(response),
                )
                if not response.strip():
                    logger.warning("MedGemma decode produced empty response")
                else:
                    logger.info("MedGemma raw output (first 500 chars): %s", response[:500])

                # Parse JSON
                logger.info("MedGemma parsing JSON response")
                # The prompt ends mid-JSON, so prepend the JSON start to the response
                prompt_json_prefix = f'{{"prediction":"{label}","confidence":{score:.2f},"morphology":"'
                full_json_attempt = prompt_json_prefix + response
                report = self._parse_json_response(full_json_attempt, case_id=case_id, score=score, label=label)

                # Validate
                logger.info("MedGemma validating report")
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
