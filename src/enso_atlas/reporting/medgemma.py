"""
MedGemma Reporter - Structured report generation from evidence.

Uses MedGemma 1.5 to generate:
- Structured JSON reports for auditing
- Human-readable tumor board summaries
- Safety-aware, grounded interpretations
"""

import inspect
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PATHOLOGY_STYLE_EXEMPLARS = """
Style exemplar A (format/tone only):
Both ovaries are replaced by poorly differentiated adenocarcinoma. The tumour is composed of complex tubulopapillary glands with multifocal solid cell groups. The tumour cells are pleomorphic, with vesicular nuclei and prominent nucleoli. Numerous mitotic figures are present, including atypical forms. Extensive necrosis is identified. The features are most in keeping with high-grade serous papillary adenocarcinoma with bilateral ovarian surface involvement.

Style exemplar B (format/tone only):
Histological sections of both ovaries show a poorly differentiated carcinoma consistent with serous adenocarcinoma, with tumour cells arranged predominantly in sheets and slit-like to rounded glandular spaces. The tumour involves the uterine serosal surface with focal extension into underlying myometrium, while the endometrial cavity is not involved. Extensive metastatic deposits are present in the omentum and periappendiceal soft tissue, and psammoma bodies are identified in omental tumour deposits.
"""


def _resolve_medgemma_model_path(model_hint: str) -> str:
    """Resolve a usable MedGemma model path/id with local-first fallback."""

    def _has_model_weights(path: Path) -> bool:
        # Common single-file formats
        if (path / "model.safetensors").exists():
            return True
        if (path / "pytorch_model.bin").exists():
            return True

        # Sharded safetensors via index json
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                weight_map = payload.get("weight_map", {})
                shard_names = (
                    sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
                )
                if not shard_names:
                    return False
                missing = [name for name in shard_names if not (path / name).exists()]
                return len(missing) == 0
            except Exception:
                return False
        return False

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        model_hint,
        os.environ.get("MEDGEMMA_MODEL_PATH", ""),
        "/app/models/medgemma-4b-it",  # Docker path
        str(repo_root / "models" / "medgemma-4b-it"),  # local repo path
        os.path.expanduser("~/med-gemma-hackathon/models/medgemma-4b-it"),  # local dev
        "models/medgemma-4b-it",  # relative path
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            p = Path(candidate).expanduser()
            if p.exists():
                if p.is_dir() and not _has_model_weights(p):
                    logger.warning(
                        "Skipping incomplete MedGemma model directory (missing weight files): %s",
                        p,
                    )
                    continue
                return str(p)
        except Exception:
            continue
    # Final fallback to HF model id (allows hub usage when local paths are missing/incomplete).
    fallback_hf_id = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
    # If the hint already looks like a non-path model id, keep it.
    if model_hint and not model_hint.startswith(("/", "./", "../", "~")):
        return model_hint
    return fallback_hf_id


def _patch_torch_autocast():
    """Patch torch.is_autocast_enabled for transformers 5.0+ compatibility.

    Newer transformers calls torch.is_autocast_enabled() with no arguments,
    but some PyTorch versions have a different signature. This patches it to work.
    """
    import torch

    original_is_autocast_enabled = getattr(torch, "is_autocast_enabled", None)

    def _autocast_fallback() -> bool:
        try:
            cpu_fn = getattr(torch, "is_autocast_cpu_enabled", None)
            cuda_fn = getattr(torch, "is_autocast_cuda_enabled", None)
            cpu_enabled = bool(cpu_fn()) if callable(cpu_fn) else False
            cuda_enabled = bool(cuda_fn()) if callable(cuda_fn) else False
            return cpu_enabled or cuda_enabled
        except Exception:
            return False

    def patched_is_autocast_enabled(*args, **kwargs):
        # Try the original function first (with provided args), then no-arg form.
        if callable(original_is_autocast_enabled):
            try:
                return bool(original_is_autocast_enabled(*args, **kwargs))
            except TypeError:
                try:
                    return bool(original_is_autocast_enabled())
                except TypeError:
                    pass
            except Exception:
                pass
        # Fallback for older/partially patched torch builds.
        return _autocast_fallback()

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
    hf_cache_dir: str = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/enso-atlas-hf"),
    )
    max_evidence_patches: int = 6  # Provide richer context for MedGemma narrative generation
    max_similar_cases: int = 2  # Include minimal cohort context without blowing up prompt size
    max_input_tokens: int = 1024  # Allow structured prompt with per-patch details
    max_output_tokens: int = 1024  # Avoid truncated JSON; prioritize complete narrative fields
    max_generation_time_s: float = (
        300.0  # CPU inference on Blackwell (no CUDA sm_121) needs ~120-180s
    )
    temperature: float = 0.0  # Deterministic output to improve JSON compliance
    top_p: float = 0.9

    def __post_init__(self):
        self.model = _resolve_medgemma_model_path(self.model)
        Path(self.hf_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)


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
                "calibration_note": {"type": "string"},
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "patch_id": {"type": "string"},
                    "morphology_description": {"type": "string"},
                    "significance": {"type": "string"},
                },
            },
        },
        "similar_examples": {"type": "array"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "suggested_next_steps": {"type": "array", "items": {"type": "string"}},
        "safety_statement": {"type": "string"},
    },
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
        if self._model is not None and self._tokenizer is not None and callable(self._tokenizer):
            return

        with self._load_lock:
            # If model is present but tokenizer state is invalid, reset and reload.
            if self._model is not None and (
                self._tokenizer is None or not callable(self._tokenizer)
            ):
                logger.warning(
                    "MedGemma in-memory state is inconsistent (model loaded, tokenizer invalid). "
                    "Resetting runtime and reloading."
                )
                self._reset_runtime()

            if (
                self._model is not None
                and self._tokenizer is not None
                and callable(self._tokenizer)
            ):
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

            # Determine device
            # Blackwell GPU (sm_121) is not officially supported by PyTorch but
            # bfloat16 matmuls work correctly via compute_90 fallback.  Enable GPU
            # for ~10-50x faster inference vs CPU.
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            logger.info(f"Loading MedGemma model on {self._device}")

            # Load tokenizer and model
            model_id = self.config.model
            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            )
            cache_dir = str(Path(self.config.hf_cache_dir).expanduser())
            common_hf_kwargs: dict[str, Any] = {"cache_dir": cache_dir}
            if hf_token:
                common_hf_kwargs["token"] = hf_token

            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **common_hf_kwargs)
            if (
                self._tokenizer is not None
                and getattr(self._tokenizer, "pad_token_id", None) is None
                and getattr(self._tokenizer, "eos_token_id", None) is not None
            ):
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Try to load processor for multimodal input
            try:
                self._processor = AutoProcessor.from_pretrained(model_id, **common_hf_kwargs)
            except Exception:
                logger.warning("Could not load processor; using text-only mode")
                self._processor = None

            # Some environments/models can yield a tokenizer object that is not callable.
            # In that case, try to recover from processor.tokenizer before failing hard.
            if self._tokenizer is None or not callable(self._tokenizer):
                logger.warning(
                    "Loaded tokenizer is not callable (%s); trying processor tokenizer fallback",
                    type(self._tokenizer),
                )
                proc_tokenizer = getattr(self._processor, "tokenizer", None)
                if proc_tokenizer is not None:
                    self._tokenizer = proc_tokenizer
                if self._tokenizer is None or not callable(self._tokenizer):
                    raise TypeError(f"Loaded tokenizer is not callable: {type(self._tokenizer)}")

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
            model_kwargs.update(common_hf_kwargs)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs,
                )
            except (TypeError, RuntimeError) as load_err:
                logger.warning(
                    f"Model load with bfloat16 failed ({load_err}), retrying with float32"
                )
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

    def _reset_runtime(self) -> None:
        """Reset runtime-loaded model/tokenizer state for a clean reload."""
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None
        self._effective_max_input_tokens = None
        self._supports_max_time = None
        self._supports_stopping_criteria = None
        self._warmup_done = False

    def _force_reload_model(self) -> None:
        """Force a full MedGemma reload regardless of current in-memory state."""
        with self._load_lock:
            self._reset_runtime()
        self._load_model()

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

    def _format_patient_context(self, patient_context: dict | None) -> str:
        """Format patient context into a clinical description."""
        if not patient_context:
            return "No patient demographics available."

        parts = []
        if patient_context.get("age"):
            parts.append(f"{patient_context['age']}-year-old")
        if patient_context.get("sex"):
            sex_full = (
                "female"
                if patient_context["sex"].upper() == "F"
                else "male"
                if patient_context["sex"].upper() == "M"
                else patient_context["sex"]
            )
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

    def _classify_patch(self, patch: dict) -> str:
        """Derive tissue category for a patch, matching the API's classify_tissue_type logic."""
        # Use explicit category/tissue_type if present
        cat = patch.get("category") or patch.get("tissue_type")
        if cat and cat != "unknown":
            return cat
        # Fallback: deterministic from patch_index (same as API)
        idx = patch.get("patch_index", 0)
        return self.TISSUE_TYPES[idx % len(self.TISSUE_TYPES)]

    def _position_from_patch(self, patch: dict) -> str:
        """Resolve a human-readable spatial position label for a patch/region."""
        explicit = str(patch.get("position_label", "") or "").strip()
        if explicit:
            return explicit

        coords = patch.get("coordinates", [])
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return f"x={int(coords[0])}, y={int(coords[1])}"

        macro_region = str(patch.get("macro_region", "") or "").strip().lower().replace("_", " ")
        if macro_region:
            return macro_region

        return "unspecified position"

    def _sanitize_region_narrative(self, text: str) -> str:
        """Remove low-level numeric metadata from region narrative text."""
        if not text:
            return ""
        cleaned = str(text)
        cleaned = re.sub(
            r"\(([^)]*(attention|coverage|n_patches|coordinates?|rank|weight|center)[^)]*)\)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(attention|coverage|n_patches|coordinates?|rank|weight|center)\s*=?\s*[-+]?\d*\.?\d+\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bat\s*x\s*=\s*[-+]?\d*\.?\d+%?\s*,\s*y\s*=\s*[-+]?\d*\.?\d+%?\s*,?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bx\s*=\s*[-+]?\d*\.?\d+%?\s*,\s*y\s*=\s*[-+]?\d*\.?\d+%?\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -;,.")
        lower = cleaned.lower()
        generic_patterns = [
            r"did not provide",
            r"not explicitly provided",
            r"morphologically aligned with",
            r"associated with the predicted classification",
            r"based on model attention analysis",
            r"consistent with the model interpretation",
            r"this macroregion",
            r"\b(the\s+)?(tumor|stroma|necrosis|inflammatory|normal|artifact)\s+(tissue\s+)?region is characterized by\b",
            r"\b(the\s+)?(tumor|stroma|necrosis|inflammatory|normal|artifact)\s+(tissue\s+)?region (contains|is dominated by)\b",
        ]
        if any(re.search(p, lower) for p in generic_patterns):
            return ""
        if len(cleaned.split()) < 7:
            return ""
        return cleaned

    @staticmethod
    def _is_non_informative_text(text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return True
        lower = cleaned.lower()
        blocked = [
            "did not provide",
            "not explicitly provided",
            "not available",
            "not provided",
            "unknown",
            "n/a",
            "insufficient information",
            "cannot determine",
        ]
        return any(b in lower for b in blocked)

    @staticmethod
    def _as_sentence(text: str) -> str:
        """Normalize free text into a single clean sentence/paragraph fragment."""
        cleaned = str(text or "")
        cleaned = re.sub(r"(?im)^\s*model interpretation\s*:\s*", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^\s*-\s*morphology overview\s*:\s*", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^\s*-\s*clinical significance\s*:\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\t")
        if not cleaned:
            return ""
        if not re.search(r"[.!?]$", cleaned):
            cleaned += "."
        return cleaned

    def _build_overview_from_evidence(self, evidence_patches: list[dict], label: str) -> str:
        tissue_counts: dict[str, int] = {}
        for patch in evidence_patches:
            dominant = patch.get("dominant_tissues")
            if isinstance(dominant, list) and dominant:
                for t in dominant[:2]:
                    tt = str(t).strip().lower()
                    if tt:
                        tissue_counts[tt] = tissue_counts.get(tt, 0) + 1
            else:
                tt = self._classify_patch(patch).strip().lower()
                if tt:
                    tissue_counts[tt] = tissue_counts.get(tt, 0) + 1

        dominant_sorted = sorted(tissue_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_tissues = [k.replace("_", " ") for k, _ in dominant_sorted[:4]]
        if top_tissues:
            if len(top_tissues) == 1:
                tissue_phrase = top_tissues[0]
            elif len(top_tissues) == 2:
                tissue_phrase = f"{top_tissues[0]} and {top_tissues[1]}"
            else:
                tissue_phrase = ", ".join(top_tissues[:-1]) + f", and {top_tissues[-1]}"
            return (
                f"Histological sections show heterogeneous morphology with components of {tissue_phrase}. "
                "Architectural and cytologic features vary across sampled regions, indicating mixed histologic composition."
            )
        return (
            "Histological sections show mixed tissue morphology with region-to-region heterogeneity. "
            "The aggregate findings support a non-uniform histopathologic pattern on this slide."
        )

    @staticmethod
    def _build_clinical_significance_from_prediction(label: str, probability: float) -> str:
        try:
            p = float(probability)
        except Exception:
            p = 0.5
        p = max(0.0, min(1.0, p))
        confidence = abs(p - 0.5) * 2.0
        if confidence >= 0.70:
            conf_txt = "high"
        elif confidence >= 0.40:
            conf_txt = "moderate"
        else:
            conf_txt = "low"
        return (
            f"The overall features are most in keeping with a {str(label).lower()} model pattern "
            f"with {conf_txt} internal confidence. "
            "This interpretation is for decision support and should be integrated with clinical history, biomarkers, and formal pathology review."
        )

    def _build_region_fallback_narrative(
        self,
        patch: dict,
        *,
        tissue_title: str,
        position_label: str,
        label: str,
    ) -> str:
        dominant = patch.get("dominant_tissues", [])
        dominant_clean: list[str] = []
        if isinstance(dominant, list):
            for t in dominant[:3]:
                s = str(t).replace("_", " ").strip().lower()
                if s:
                    dominant_clean.append(s)
        if not dominant_clean:
            dominant_clean = [tissue_title.lower()]

        if len(dominant_clean) == 1:
            dom_phrase = dominant_clean[0]
        elif len(dominant_clean) == 2:
            dom_phrase = f"{dominant_clean[0]} and {dominant_clean[1]}"
        else:
            dom_phrase = ", ".join(dominant_clean[:-1]) + f", and {dominant_clean[-1]}"

        primary = dominant_clean[0] if dominant_clean else tissue_title.lower()
        coverage = patch.get("coverage_pct")
        coverage_txt = ""
        if isinstance(coverage, (int, float)):
            cov = float(coverage)
            if cov < 5.0:
                coverage_txt = f"This is a focal component (about {cov:.1f}% of analyzed patches)."
            elif cov < 15.0:
                coverage_txt = (
                    f"This represents a limited component (about {cov:.1f}% of analyzed patches)."
                )
            elif cov < 35.0:
                coverage_txt = f"This represents a substantial regional component (about {cov:.1f}% of analyzed patches)."
            else:
                coverage_txt = (
                    f"This is a dominant component (about {cov:.1f}% of analyzed patches)."
                )

        tissue_specific: dict[str, str] = {
            "tumor": "This region is enriched for viable tumor morphology with atypical epithelial architecture.",
            "stroma": "This region is stroma-rich and shows tumor-associated stromal remodeling patterns.",
            "necrosis": "This region shows necrotic and degenerative morphology consistent with non-viable tissue burden.",
            "inflammatory": "This region is enriched for inflammatory/immune infiltrative morphology.",
            "artifact": "This region contains prominent slide artifact and mixed tissue context, which reduces interpretability confidence.",
            "normal": "This region is closer to non-neoplastic background morphology than overt malignant architecture.",
        }
        opener = tissue_specific.get(
            primary, f"This region is morphologically dominated by {dom_phrase} patterns."
        )

        if len(dominant_clean) >= 3:
            heterogeneity_txt = f" It also contains {dominant_clean[1]} and {dominant_clean[2]} components, indicating marked intraregional heterogeneity."
        elif len(dominant_clean) == 2:
            heterogeneity_txt = (
                f" It also includes {dominant_clean[1]} morphology, indicating mixed composition."
            )
        else:
            heterogeneity_txt = " Morphology in this cluster is relatively homogeneous."

        label_txt = str(label).lower()
        contribution_txt = f" In aggregate with other macroregions, this pattern contributes to the model's {label_txt} output."

        parts = [opener, heterogeneity_txt]
        if coverage_txt:
            parts.append(f" {coverage_txt}")
        parts.append(contribution_txt)
        return "".join(parts).strip()

    @staticmethod
    def _region_narrative_conflicts_tissue(text: str, tissue_title: str, patch: dict) -> bool:
        if not text:
            return True
        allowed = {str(tissue_title).strip().lower()}
        dominant = patch.get("dominant_tissues", [])
        if isinstance(dominant, list):
            for t in dominant:
                s = str(t).replace("_", " ").strip().lower()
                if s:
                    allowed.add(s)
        tissue_match = re.search(
            r"\b(the\s+)?(tumor|stroma|necrosis|inflammatory|normal|artifact)(\s+tissue)?\s+region\b",
            str(text).strip().lower(),
        )
        if tissue_match:
            mentioned = str(tissue_match.group(2)).strip().lower()
            if mentioned not in allowed:
                return True
        return False

    def _build_prompt(
        self,
        evidence_patches: list[dict],
        score: float,
        label: str,
        similar_cases: list[dict],
        case_id: str = "unknown",
        patient_context: dict | None = None,
        cancer_type: str = "Cancer",
    ) -> str:
        """Build a rich but bounded prompt that requests structured MedGemma narrative fields."""

        patient_summary = self._format_patient_context(patient_context)

        evidence_lines = []
        for i, patch in enumerate(evidence_patches, 1):
            rank = patch.get("rank", i)
            patch_index = patch.get("patch_index", i - 1)
            patch_id = str(patch.get("patch_id") or f"patch_{rank}_idx{patch_index}")
            attn = float(patch.get("attention_weight", 0.0))
            attn_peak = patch.get("attention_peak")
            coords = patch.get("coordinates", [])
            tissue = self._classify_patch(patch)
            region_name = str(patch.get("macro_region", "") or "").strip().replace("_", " ")
            dominant_tissues = patch.get("dominant_tissues", [])
            if isinstance(dominant_tissues, list):
                dominant_tissues = [str(t).strip() for t in dominant_tissues if str(t).strip()]
            else:
                dominant_tissues = []
            coverage = patch.get("coverage_pct")
            n_patches = patch.get("n_patches")
            coord_str = (
                f"({int(coords[0])}, {int(coords[1])})"
                if isinstance(coords, (list, tuple)) and len(coords) >= 2
                else "unknown"
            )
            region_prefix = f"region={region_name}, " if region_name else ""
            dominant_part = (
                f", dominant_tissues={', '.join(dominant_tissues)}" if dominant_tissues else ""
            )
            coverage_part = (
                f", coverage={float(coverage):.1f}%" if isinstance(coverage, (int, float)) else ""
            )
            n_part = f", n_patches={int(n_patches)}" if isinstance(n_patches, (int, float)) else ""
            peak_part = (
                f", attention_peak={float(attn_peak):.4f}"
                if isinstance(attn_peak, (int, float))
                else ""
            )
            evidence_lines.append(
                f"- {patch_id}: {region_prefix}tissue={tissue}, attention_mean={attn:.4f}{peak_part}, "
                f"coordinates={coord_str}{coverage_part}{n_part}{dominant_part}"
            )
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- none"
        n_regions = max(1, len(evidence_patches))

        similar_lines = []
        for i, s in enumerate(similar_cases, 1):
            if not isinstance(s, dict):
                continue
            meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
            sid = str(s.get("slide_id") or meta.get("slide_id") or f"case_{i}")
            slabel = str(meta.get("label") or s.get("label") or "unknown")
            dist = s.get("distance")
            if isinstance(dist, (int, float)):
                similar_lines.append(f"- {sid}: label={slabel}, distance={float(dist):.4f}")
            else:
                similar_lines.append(f"- {sid}: label={slabel}")
        similar_block = "\n".join(similar_lines) if similar_lines else "- none"

        prompt = f"""# CONTEXT #
You are an expert computational pathologist and AI diagnostic assistant operating in a high-stakes research environment. You are evaluating Whole Slide Image (WSI) H&E patches of <cancer_type> to predict platinum sensitivity. You are provided with a Multiple Instance Learning (MIL) prediction, patient clinical context, top morphological evidence patches, and similar reference cases from a research database.

# OBJECTIVE #
Synthesize the provided MIL prediction, patient context, and morphological evidence patches into a comprehensive, biologically plausible histopathology report. Your goal is to explain the morphological basis for the predicted platinum sensitivity label based strictly on the provided evidence.

# STYLE #
Write in formal, academic histopathology report prose. Use standard histological terminology (e.g., "Histological sections demonstrate...", "The morphologic features are most in keeping with..."). Emulate the writing style of the provided <pathology_style_exemplars>, but do not copy them verbatim.

# TONE #
Highly clinical, objective, cautious, and uncertainty-aware. Because this is for research use only, avoid definitive diagnostic declarations. Be precise, analytical, and scientifically rigorous.

# AUDIENCE #
Translational researchers, computational pathologists, and gynecologic oncologists investigating platinum resistance in ovarian cancer.

# RESPONSE #
Return ONLY a single, valid JSON object. Do not include markdown formatting, code fences (like ```json), or extra commentary.

Follow these strict rules:
1. Keep "prediction" and "confidence" numerically consistent with the provided MIL values.
2. Tie findings explicitly to morphology and the provided evidence patches.
3. Provide exactly {n_regions} items in "key_findings" and exactly {n_regions} items in "patch_significance" (one narrative item per evidence region, in the exact same order as the evidence list).
4. Keep each list item concise and non-redundant.

CRITICAL NEGATIVE CONSTRAINTS:
- Do NOT include attention weights, coverage values, ranks, coordinates, or any other raw numeric metadata in the text.
- Do NOT mention model mechanics (attention, clustering, embeddings, macroregions, patches, MIL) in the narrative text.
- Do NOT invent unsupported anatomic spread, grades, biomarker/IHC results, or staging details.
- Do NOT output placeholders such as "did not provide", "not available", or "unknown".

EXPECTED JSON SCHEMA:
{{
  "morphological_reasoning": "Provide a step-by-step visual analysis of the provided patches. Think logically about the cellular features before generating the rest of the report.",
  "prediction": "{label}",
  "confidence": {score:.4f},
  "morphology_overview": "2-3 sentence slide-level morphology synthesis",
  "clinical_significance": "1-2 sentence clinical interpretation with uncertainty-aware wording",
  "key_findings": ["region finding 1", "region finding 2"],
  "patch_significance": ["region significance 1", "region significance 2"],
  "recommendation": "one primary recommendation",
  "suggested_next_steps": ["step 1", "step 2", "step 3"]
}}

# INPUT DATA #
<case_id>{case_id}</case_id>
<cancer_type>{cancer_type}</cancer_type>
<mil_prediction>{label}</mil_prediction>
<mil_confidence>{score:.4f}</mil_confidence>
<patient_context>{patient_summary}</patient_context>

<evidence_patches>
{evidence_block}
</evidence_patches>

<similar_cases>
{similar_block}
</similar_cases>

<pathology_style_exemplars>
{PATHOLOGY_STYLE_EXEMPLARS}
</pathology_style_exemplars>
"""
        return prompt

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for the prompt."""
        if self._tokenizer is None:
            return max(1, len(prompt.split()))
        encode_fn = getattr(self._tokenizer, "encode", None)
        if callable(encode_fn):
            try:
                return len(encode_fn(prompt, add_special_tokens=False))
            except Exception:
                pass
        try:
            tokenized = self._tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = tokenized.get("input_ids", [])
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return len(input_ids) if isinstance(input_ids, list) else max(1, len(prompt.split()))
        except Exception:
            return max(1, len(prompt.split()))

    def _parse_json_response(
        self,
        response: str,
        case_id: str = "unknown",
        score: float = 0.0,
        label: str = "unknown",
        evidence_patches: list[dict] | None = None,
        cancer_type: str = "Cancer",
    ) -> dict:
        """Extract and parse JSON from model response, mapping simplified output to full report structure.

        Evidence entries are ALWAYS built from the real evidence_patches (with attention
        weights and coordinates computed upstream).  MedGemma output is used only to
        enrich those entries with morphology descriptions and clinical significance.
        If MedGemma produces nothing useful, evidence entries still contain the core
        patch data so the caller never sees evidence=0.
        """
        if evidence_patches is None:
            evidence_patches = []

        # --- Extract JSON from the model response ---------------------------
        # Extract the FIRST JSON object from markdown code blocks
        # Model sometimes repeats the JSON multiple times in fences
        code_block_match = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", response)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            # Fallback: find first JSON object directly
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
            json_str = json_match.group() if json_match else None

        # Legacy fallback for any JSON
        if not json_str:
            cleaned = response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            json_match = re.search(r"\{[\s\S]*?\}", cleaned)
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
                    '"}',  # truncated string value
                    '"]',  # truncated array
                    '"]}',  # truncated array in object
                    '"}]}',  # truncated string in array in object
                    '"]}}}',  # deeply nested
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
                    for field in [
                        "prediction",
                        "morphology",
                        "morphology_overview",
                        "morphology_description",
                        "clinical_significance",
                        "recommendation",
                    ]:
                        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', json_str)
                        if m:
                            parsed[field] = m.group(1)
                    # Extract list-like fields
                    for list_field in [
                        "key_findings",
                        "patch_significance",
                        "suggested_next_steps",
                    ]:
                        lf_match = re.search(rf'"{list_field}"\s*:\s*\[(.*)', json_str)
                        if lf_match:
                            entries = re.findall(r'"([^"]+)"', lf_match.group(1))
                            if entries:
                                parsed[list_field] = entries
                    if parsed:
                        logger.info("Extracted %d fields via regex fallback", len(parsed))
                    else:
                        logger.warning("Failed to parse JSON even after repair and regex attempts")

        if parsed is None:
            parsed = {}

        def _coerce_string_list(value: Any) -> list[str]:
            if isinstance(value, list):
                out = []
                for item in value:
                    text = str(item).strip()
                    if text:
                        out.append(text)
                return out
            if isinstance(value, str):
                chunks = re.split(r"[;\n]+", value)
                return [c.strip(" -\t") for c in chunks if c and c.strip(" -\t")]
            return []

        def _extract_labeled_field(text: str, labels: list[str]) -> str:
            if not text:
                return ""
            for lbl in labels:
                m = re.search(
                    rf"{re.escape(lbl)}\s*[:\-]\s*(.+?)(?:\n[A-Za-z _]+[:\-]|$)",
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m:
                    candidate = str(m.group(1)).strip().strip('"').strip()
                    candidate = re.sub(r"\s+", " ", candidate)
                    if candidate:
                        return candidate
            return ""

        def _clean_text_list(items: list[str]) -> list[str]:
            cleaned_items: list[str] = []
            for item in items:
                s = self._sanitize_region_narrative(item)
                low = s.lower()
                if not s:
                    continue
                if "should be" in low and ("morphology" in low or "clinical" in low):
                    continue
                if low.startswith("the ") and " should " in low:
                    continue
                cleaned_items.append(s)
            return cleaned_items

        raw_response_text = str(response or "")

        # --- Extract MedGemma enrichment fields ------------------------------
        key_findings = _clean_text_list(_coerce_string_list(parsed.get("key_findings", [])))
        patch_significance = _clean_text_list(
            _coerce_string_list(parsed.get("patch_significance", []))
        )
        next_steps = _coerce_string_list(parsed.get("suggested_next_steps", []))
        recommendation = str(parsed.get("recommendation", "") or "").strip()

        # Accept both legacy and current morphology fields.
        # Some model outputs use camelCase or relaxed labels.
        def _first_nonempty(*vals: Any) -> str:
            for v in vals:
                s = str(v or "").strip()
                if s:
                    return s
            return ""

        morphology_desc = _first_nonempty(
            parsed.get("morphology_overview"),
            parsed.get("morphologyOverview"),
            parsed.get("morphology_description"),
            parsed.get("morphologyDescription"),
            parsed.get("morphology"),
            parsed.get("overview"),
        )
        clinical_sig = _first_nonempty(
            parsed.get("clinical_significance"),
            parsed.get("clinicalSignificance"),
            parsed.get("clinical significance"),
            parsed.get("clinical_interpretation"),
            parsed.get("interpretation"),
        )
        if not morphology_desc:
            morphology_desc = _extract_labeled_field(
                raw_response_text,
                [
                    "morphology_overview",
                    "morphology overview",
                    "morphology description",
                    "morphology",
                ],
            )
        if not clinical_sig:
            clinical_sig = _extract_labeled_field(
                raw_response_text,
                [
                    "clinical_significance",
                    "clinical significance",
                    "clinical interpretation",
                    "interpretation",
                ],
            )
        if self._is_non_informative_text(morphology_desc):
            morphology_desc = ""
        if self._is_non_informative_text(clinical_sig):
            clinical_sig = ""

        raw_conf = parsed.get("confidence", score)
        try:
            parsed_confidence = float(raw_conf)
        except (TypeError, ValueError):
            parsed_confidence = float(score)
        parsed_confidence = max(0.0, min(1.0, parsed_confidence))

        if not morphology_desc:
            morphology_desc = self._build_overview_from_evidence(evidence_patches, label)
        if not clinical_sig:
            clinical_sig = self._build_clinical_significance_from_prediction(
                label, parsed_confidence
            )

        next_steps_final = list(next_steps)
        if recommendation and recommendation not in next_steps_final:
            next_steps_final.insert(0, recommendation)
        if not next_steps_final:
            next_steps_final = [
                "Correlate regional morphology with patient history, laboratory data, and molecular findings.",
                "Review with multidisciplinary pathology and oncology teams before treatment decisions.",
            ]

        def _normalize_guideline_references(value: Any) -> list[dict[str, str]]:
            refs: list[dict[str, str]] = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        source = str(item.get("source", "")).strip()
                        section = str(item.get("section", "")).strip()
                        rec = str(item.get("recommendation", "")).strip()
                        url = str(item.get("url", "")).strip()
                        if source or section or rec:
                            entry: dict[str, str] = {
                                "source": source or "Clinical guideline",
                                "section": section or "General",
                                "recommendation": rec
                                or "Correlate with multidisciplinary evaluation.",
                            }
                            if url:
                                entry["url"] = url
                            refs.append(entry)
            return refs

        def _normalize_decision_support(
            raw: Any,
            *,
            fallback_confidence_score: float,
            fallback_primary_recommendation: str,
            fallback_rationale: list[str],
            fallback_workup: list[str],
            fallback_interpretation: str,
            fallback_caveat: str,
        ) -> dict[str, Any] | None:
            raw_ds = raw if isinstance(raw, dict) else {}
            if not raw_ds and not fallback_primary_recommendation:
                return None

            def _norm_level(value: Any, allowed: set[str], default: str) -> str:
                text = str(value or "").strip().lower()
                return text if text in allowed else default

            confidence_score = raw_ds.get("confidence_score")
            try:
                confidence_score_val = float(confidence_score)
            except (TypeError, ValueError):
                confidence_score_val = fallback_confidence_score
            confidence_score_val = max(0.0, min(1.0, confidence_score_val))

            confidence_level_default = (
                "high"
                if confidence_score_val >= 0.70
                else "moderate"
                if confidence_score_val >= 0.35
                else "low"
            )
            confidence_level = _norm_level(
                raw_ds.get("confidence_level"),
                {"high", "moderate", "low"},
                confidence_level_default,
            )

            risk_level_default = (
                "high_confidence"
                if confidence_score_val >= 0.70
                else "moderate_confidence"
                if confidence_score_val >= 0.45
                else "low_confidence"
            )
            risk_level = _norm_level(
                raw_ds.get("risk_level"),
                {"high_confidence", "moderate_confidence", "low_confidence", "inconclusive"},
                risk_level_default,
            )

            primary_recommendation = str(
                raw_ds.get("primary_recommendation") or fallback_primary_recommendation
            ).strip()
            if not primary_recommendation:
                return None
            supporting_rationale = (
                _coerce_string_list(raw_ds.get("supporting_rationale", [])) or fallback_rationale
            )
            alternative_considerations = _coerce_string_list(
                raw_ds.get("alternative_considerations", [])
            )
            guideline_references = _normalize_guideline_references(
                raw_ds.get("guideline_references", [])
            )
            uncertainty_statement = str(raw_ds.get("uncertainty_statement") or "").strip()
            if not uncertainty_statement and fallback_interpretation:
                uncertainty_statement = (
                    f"Confidence is {confidence_level}; integrate morphology with clinical and molecular context. "
                    f"{fallback_interpretation}"
                )
            quality_warnings = _coerce_string_list(raw_ds.get("quality_warnings", []))
            suggested_workup = (
                _coerce_string_list(raw_ds.get("suggested_workup", [])) or fallback_workup
            )
            interpretation_note = str(
                raw_ds.get("interpretation_note") or fallback_interpretation
            ).strip()
            caveat = str(raw_ds.get("caveat") or fallback_caveat).strip()

            return {
                "source_model": "medgemma",
                "risk_level": risk_level,
                "confidence_level": confidence_level,
                "confidence_score": confidence_score_val,
                "primary_recommendation": primary_recommendation,
                "supporting_rationale": supporting_rationale,
                "alternative_considerations": alternative_considerations,
                "guideline_references": guideline_references,
                "uncertainty_statement": uncertainty_statement,
                "quality_warnings": quality_warnings,
                "suggested_workup": suggested_workup,
                "interpretation_note": interpretation_note,
                "caveat": caveat,
            }

        confidence_score_for_decision = max(0.0, min(1.0, abs(parsed_confidence - 0.5) * 2.0))
        fallback_primary = recommendation or (next_steps_final[0] if next_steps_final else "")
        fallback_rationale = key_findings[:3]
        fallback_workup = next_steps_final[:4]
        fallback_interpretation = clinical_sig
        fallback_caveat = "Research-use output; not a standalone diagnostic recommendation."
        decision_support_payload = _normalize_decision_support(
            parsed.get("decision_support"),
            fallback_confidence_score=confidence_score_for_decision,
            fallback_primary_recommendation=fallback_primary,
            fallback_rationale=fallback_rationale,
            fallback_workup=fallback_workup,
            fallback_interpretation=fallback_interpretation,
            fallback_caveat=fallback_caveat,
        )

        parsed_label_raw = str(parsed.get("prediction", "") or "").strip()
        parsed_label = (
            parsed_label_raw
            if parsed_label_raw and not self._is_non_informative_text(parsed_label_raw)
            else str(label)
        )

        # --- Build evidence from REAL patches, enriched by MedGemma ----------
        # UI requirement: morphology_description should be a concise region title,
        # while significance carries the narrative interpretation.
        evidence_entries = []
        for i, patch in enumerate(evidence_patches):
            rank = patch.get("rank", i + 1)
            patch_index = patch.get("patch_index", i)
            patch_id = str(patch.get("patch_id") or f"patch_{rank}_idx{patch_index}")
            coords = patch.get("coordinates", [])
            tissue = self._classify_patch(patch)
            tissue_title = str(tissue).replace("_", " ").strip().title() or "Morphology"
            position_label = self._position_from_patch(patch)
            macro_region = str(patch.get("macro_region", "") or "").strip()

            patch_morphology = f"{tissue_title} tissue region ({position_label})"

            patch_sig_candidate = ""
            if i < len(patch_significance) and patch_significance[i]:
                patch_sig_candidate = patch_significance[i]
            elif i < len(key_findings) and key_findings[i]:
                patch_sig_candidate = key_findings[i]

            patch_sig = self._sanitize_region_narrative(patch_sig_candidate)
            if patch_sig and clinical_sig and patch_sig.lower() in clinical_sig.lower():
                patch_sig = ""
            fallback_sig = self._build_region_fallback_narrative(
                patch,
                tissue_title=tissue_title,
                position_label=position_label,
                label=label,
            )
            if self._region_narrative_conflicts_tissue(patch_sig, tissue_title, patch):
                patch_sig = fallback_sig
            elif patch_sig:
                patch_sig = f"{patch_sig.rstrip('.')} {fallback_sig}"
            else:
                patch_sig = fallback_sig

            coords_out = [0, 0]
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    coords_out = [int(coords[0]), int(coords[1])]
                except Exception:
                    coords_out = [0, 0]

            evidence_entries.append(
                {
                    "patch_id": patch_id,
                    "coordinates": coords_out,
                    "macro_region": macro_region,
                    "position_label": position_label,
                    "morphology_description": patch_morphology,
                    "significance": patch_sig,
                }
            )

        report_payload: dict[str, Any] = {
            "case_id": case_id,
            "task": f"{cancer_type} treatment response prediction from H&E histopathology",
            "model_output": {
                "label": parsed_label,
                "probability": parsed_confidence,
                "calibration_note": "Model probability, not clinical certainty. Requires external validation.",
                "clinical_significance": clinical_sig,
            },
            "model_generated_overview": morphology_desc,
            "evidence": evidence_entries,
            "limitations": [
                "AI prediction requires pathologist validation",
                "Based on H&E morphology only -- no IHC or molecular data",
                "Training cohort may not represent all patient populations",
            ],
            "suggested_next_steps": next_steps_final,
            "safety_statement": "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists.",
        }
        if decision_support_payload is not None:
            report_payload["decision_support"] = decision_support_payload
        return report_payload

    def _validate_report(self, report: dict) -> bool:
        """Validate report against schema."""
        required_fields = [
            "case_id",
            "task",
            "model_output",
            "evidence",
            "limitations",
            "safety_statement",
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
        evidence_patches: list[dict],
        score: float,
        label: str,
        similar_cases: list[dict],
        case_id: str = "unknown",
        max_retries: int = 2,
        patient_context: dict | None = None,
        cancer_type: str = "Cancer",
    ) -> dict:
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

        # Self-heal occasional runtime corruption where tokenizer/model references
        # become invalid between requests.
        if self._tokenizer is None or not callable(self._tokenizer):
            logger.warning(
                "MedGemma tokenizer is invalid before generation; forcing model reload once."
            )
            self._force_reload_model()
        if self._tokenizer is None or not callable(self._tokenizer):
            raise RuntimeError("MedGemma tokenizer is not initialized or is not callable")
        if self._model is None or not callable(getattr(self._model, "generate", None)):
            logger.warning(
                "MedGemma model generate() is unavailable before generation; forcing model reload once."
            )
            self._force_reload_model()
        if self._model is None or not callable(getattr(self._model, "generate", None)):
            raise RuntimeError("MedGemma model is not initialized or generate() is unavailable")

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
            cancer_type,
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
                    cancer_type,
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
                cancer_type,
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
                    hit_time_limit = max_time is not None and gen_elapsed >= max_time - 0.5
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
                decode_ids = output_ids[inputs["input_ids"].shape[1] :]
                decode_fn = getattr(self._tokenizer, "decode", None)
                if callable(decode_fn):
                    response = decode_fn(
                        decode_ids,
                        skip_special_tokens=True,
                    )
                else:
                    batch_decode_fn = getattr(self._tokenizer, "batch_decode", None)
                    if not callable(batch_decode_fn):
                        raise RuntimeError("Tokenizer decode functions are unavailable")
                    response_list = batch_decode_fn([decode_ids], skip_special_tokens=True)
                    response = response_list[0] if response_list else ""
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
                report = self._parse_json_response(
                    response,
                    case_id=case_id,
                    score=score,
                    label=label,
                    evidence_patches=evidence_patches[:evidence_limit],
                    cancer_type=cancer_type,
                )

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
                logger.exception("Report generation error (attempt %d)", attempt + 1)
                # Auto-recover from sporadic runtime corruption where a callable becomes None.
                if (
                    isinstance(e, TypeError)
                    and "NoneType" in str(e)
                    and "callable" in str(e)
                    and attempt < max_retries
                ):
                    try:
                        logger.warning(
                            "Detected NoneType-callable MedGemma failure, forcing model reload and retry"
                        )
                        self._force_reload_model()
                        continue
                    except Exception:
                        logger.exception("MedGemma force-reload failed")

        # Return fallback report
        return self._create_fallback_report(case_id, score, label)

    def _create_fallback_report(
        self,
        case_id: str,
        score: float,
        label: str,
    ) -> dict:
        """Create a fallback report when generation fails."""
        return {
            "case_id": case_id,
            "task": "Treatment response prediction from H&E histopathology",
            "model_output": {
                "label": label,
                "probability": score,
                "calibration_note": "Automated report generation failed. Manual interpretation required.",
            },
            "evidence": [],
            "limitations": [
                "Automated report generation failed",
                "Evidence interpretation not available",
                "Manual pathology review required",
            ],
            "suggested_next_steps": [
                "Manual review of evidence patches",
                "Consultation with pathology team",
            ],
            "safety_statement": "This is a research tool. Report generation encountered errors. All findings require manual validation by qualified pathologists.",
        }

    @staticmethod
    def _is_fallback_report_payload(report: dict[str, Any]) -> bool:
        """Detect internal fallback payloads emitted after generation failure."""
        note = (
            report.get("model_output", {}).get("calibration_note", "")
            if isinstance(report, dict)
            else ""
        )
        return isinstance(note, str) and "Automated report generation failed" in note

    def generate_report(
        self,
        evidence_patches: list[dict],
        score: float,
        label: str,
        similar_cases: list[dict],
        case_id: str = "unknown",
        patient_context: dict | None = None,
        cancer_type: str = "Cancer",
    ) -> dict:
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
            cancer_type=cancer_type,
        )
        if self._is_fallback_report_payload(structured):
            raise RuntimeError("MedGemma returned fallback payload")

        # Generate human-readable summary
        summary = self.generate_summary(structured)

        return {
            "structured": structured,
            "summary": summary,
        }

    def generate_summary(self, report: dict) -> str:
        """
        Generate a human-readable summary from structured report.

        Args:
            report: Structured report dictionary

        Returns:
            Formatted text summary
        """
        model_output = report.get("model_output", {})
        overview = str(report.get("model_generated_overview", "") or "").strip()
        clinical_sig = str(model_output.get("clinical_significance", "") or "").strip()
        evidence = report.get("evidence", []) if isinstance(report.get("evidence"), list) else []
        label_txt = str(model_output.get("label", "unknown") or "unknown").strip().lower()
        try:
            prob = float(model_output.get("probability", 0.0))
        except Exception:
            prob = 0.0

        if self._is_non_informative_text(overview):
            overview = ""
        if self._is_non_informative_text(clinical_sig):
            clinical_sig = ""

        if not overview:
            overview = self._build_overview_from_evidence(evidence, label_txt)
        if not clinical_sig:
            clinical_sig = self._build_clinical_significance_from_prediction(label_txt, prob)
        overview = self._as_sentence(overview)
        clinical_sig = self._as_sentence(clinical_sig)

        if overview and not re.match(r"(?i)^(histologic|histological)\s+sections", overview):
            lowered = overview[0].lower() + overview[1:] if len(overview) > 1 else overview.lower()
            overview = f"Histological sections show {lowered}"

        if label_txt and label_txt not in clinical_sig.lower():
            clinical_sig = (
                f"{clinical_sig.rstrip('.')} "
                f"The overall features are most in keeping with a {label_txt} model pattern."
            )
            clinical_sig = self._as_sentence(clinical_sig)

        return f"MEDGEMMA SUMMARY:\n{overview}\n\n{clinical_sig}"
