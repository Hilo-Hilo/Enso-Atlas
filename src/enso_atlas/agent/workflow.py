"""
Enso Atlas Agent Workflow - Multi-step agentic analysis with reasoning.

This module implements an AI agent that performs comprehensive slide analysis
through a multi-step workflow with visible reasoning, retrieval, and report generation.

Targets the "Agentic Workflows" special prize track for the MedGemma hackathon.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import asyncio
import logging
import json
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class AgentStep(str, Enum):
    """Steps in the agent workflow."""
    INITIALIZE = "initialize"
    ANALYZE = "analyze"
    RETRIEVE = "retrieve"
    COMPARE = "compare"
    SEMANTIC_SEARCH = "semantic_search"
    REASON = "reason"
    REPORT = "report"
    COMPLETE = "complete"
    ERROR = "error"


class StepStatus(str, Enum):
    """Status of each workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class StepResult:
    """Result from a single workflow step."""
    step: AgentStep
    status: StepStatus
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    duration_ms: float = 0.0


@dataclass
class AnalysisContext:
    """Context for the current slide being analyzed."""
    slide_id: str
    clinical_context: str = ""
    questions: List[str] = field(default_factory=list)
    
    # Populated during analysis
    embeddings: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    n_patches: int = 0
    

@dataclass 
class AgentState:
    """
    State tracking for the agent workflow.
    
    Maintains conversation history, analysis results, and enables
    multi-turn follow-up questions.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    current_step: AgentStep = AgentStep.INITIALIZE
    
    # Analysis context
    context: Optional[AnalysisContext] = None
    
    # Results from each step
    step_results: Dict[AgentStep, StepResult] = field(default_factory=dict)
    
    # Model predictions
    predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Similar cases found
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evidence patches
    top_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Semantic search results (MedSigLIP)
    semantic_search_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Generated report
    report: Optional[Dict[str, Any]] = None
    
    # Conversation history for follow-up
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Reasoning chain for explainability
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Error tracking
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Final result from the agent workflow."""
    session_id: str
    slide_id: str
    success: bool
    predictions: Dict[str, Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    top_evidence: List[Dict[str, Any]]
    semantic_search_results: Dict[str, List[Dict[str, Any]]]
    report: Optional[Dict[str, Any]]
    reasoning_chain: List[str]
    total_duration_ms: float
    error: Optional[str] = None


class AgentWorkflow:
    """
    Multi-step AI agent for comprehensive slide analysis.
    
    Workflow:
    1. ANALYZE - Run multi-model TransMIL predictions
    2. RETRIEVE - Find similar cases from reference database
    3. COMPARE - Compare current slide with similar cases
    4. REASON - Generate evidence-based reasoning (optionally with MedGemma)
    5. REPORT - Produce final structured report
    
    Each step streams intermediate results via Server-Sent Events (SSE).
    """
    
    def __init__(
        self,
        embeddings_dir: Path,
        multi_model_inference: Any = None,
        evidence_generator: Any = None,
        medgemma_reporter: Any = None,
        medsiglip_embedder: Any = None,
        slide_labels: Dict[str, str] = None,
        slide_mean_index: Any = None,
        slide_mean_ids: List[str] = None,
        slide_mean_meta: Dict[str, Dict] = None,
    ):
        self.embeddings_dir = embeddings_dir
        self.multi_model_inference = multi_model_inference
        self.evidence_generator = evidence_generator
        self.medgemma_reporter = medgemma_reporter
        self.medsiglip_embedder = medsiglip_embedder
        self.slide_labels = slide_labels or {}
        self.slide_mean_index = slide_mean_index
        self.slide_mean_ids = slide_mean_ids or []
        self.slide_mean_meta = slide_mean_meta or {}
        
        # In-memory session storage
        self._sessions: Dict[str, AgentState] = {}
        
    def get_session(self, session_id: str) -> Optional[AgentState]:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)
    
    def create_session(
        self,
        slide_id: str,
        clinical_context: str = "",
        questions: List[str] = None,
    ) -> AgentState:
        """Create a new agent session."""
        state = AgentState(
            context=AnalysisContext(
                slide_id=slide_id,
                clinical_context=clinical_context,
                questions=questions or [],
            )
        )
        self._sessions[state.session_id] = state
        logger.info(f"Created agent session {state.session_id} for slide {slide_id}")
        return state
    
    async def run_workflow(
        self,
        slide_id: str,
        clinical_context: str = "",
        questions: List[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StepResult]:
        """
        Run the full agent workflow with streaming results.
        
        Yields StepResult objects for each step, allowing real-time
        progress updates to the frontend.
        """
        import time
        start_time = time.time()
        
        # Create or get session
        if session_id and session_id in self._sessions:
            state = self._sessions[session_id]
        else:
            state = self.create_session(slide_id, clinical_context, questions)
        
        try:
            # Step 1: Initialize and load embeddings
            yield await self._step_initialize(state)
            if state.error:
                return
            
            # Step 2: Run multi-model analysis
            yield await self._step_analyze(state)
            if state.error:
                return
            
            # Step 3: Retrieve similar cases
            yield await self._step_retrieve(state)
            
            # Step 4: Semantic search via MedSigLIP
            yield await self._step_semantic_search(state)
            
            # Step 5: Compare with similar cases
            yield await self._step_compare(state)
            
            # Step 6: Generate reasoning
            yield await self._step_reason(state)
            
            # Step 7: Generate final report
            yield await self._step_report(state)
            
            # Complete
            total_duration = (time.time() - start_time) * 1000
            yield StepResult(
                step=AgentStep.COMPLETE,
                status=StepStatus.COMPLETE,
                message=f"Analysis complete in {total_duration/1000:.1f} seconds",
                data={
                    "session_id": state.session_id,
                    "total_duration_ms": total_duration,
                },
            )
            
        except Exception as e:
            logger.exception(f"Agent workflow error: {e}")
            state.error = str(e)
            yield StepResult(
                step=AgentStep.ERROR,
                status=StepStatus.ERROR,
                message=f"Workflow failed: {str(e)}",
                data={"error": str(e)},
            )
    
    async def _step_initialize(self, state: AgentState) -> StepResult:
        """Initialize: Load embeddings and validate slide."""
        import time
        start = time.time()
        
        state.current_step = AgentStep.INITIALIZE
        
        slide_id = state.context.slide_id
        emb_path = self.embeddings_dir / f"{slide_id}.npy"
        coord_path = self.embeddings_dir / f"{slide_id}_coords.npy"
        
        if not emb_path.exists():
            state.error = f"Slide {slide_id} not found"
            return StepResult(
                step=AgentStep.INITIALIZE,
                status=StepStatus.ERROR,
                message=f"Slide {slide_id} not found in embeddings database",
                reasoning="Cannot proceed without pre-computed patch embeddings.",
            )
        
        # Load embeddings
        state.context.embeddings = np.load(emb_path)
        state.context.n_patches = len(state.context.embeddings)
        
        if coord_path.exists():
            state.context.coordinates = np.load(coord_path)
        
        duration = (time.time() - start) * 1000
        
        result = StepResult(
            step=AgentStep.INITIALIZE,
            status=StepStatus.COMPLETE,
            message=f"Loaded {state.context.n_patches:,} patches for slide {slide_id}",
            data={
                "slide_id": slide_id,
                "n_patches": state.context.n_patches,
                "has_coordinates": state.context.coordinates is not None,
                "clinical_context": state.context.clinical_context,
                "questions": state.context.questions,
            },
            reasoning=f"Successfully loaded pre-computed Path Foundation embeddings. "
                     f"The slide has {state.context.n_patches:,} tissue patches for analysis.",
            duration_ms=duration,
        )
        state.step_results[AgentStep.INITIALIZE] = result
        return result
    
    async def _step_analyze(self, state: AgentState) -> StepResult:
        """Analyze: Run multi-model TransMIL predictions."""
        import time
        start = time.time()
        
        state.current_step = AgentStep.ANALYZE
        
        if self.multi_model_inference is None:
            # Fallback to basic analysis message
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.ANALYZE,
                status=StepStatus.SKIPPED,
                message="Multi-model inference not available",
                reasoning="Using placeholder predictions. Install multi-model module for full analysis.",
                duration_ms=duration,
            )
            state.step_results[AgentStep.ANALYZE] = result
            return result
        
        try:
            # Run multi-model inference
            results = self.multi_model_inference.predict_all(
                state.context.embeddings,
                return_attention=True,
            )
            
            state.predictions = results["predictions"]
            
            # Extract top evidence patches from platinum sensitivity model (primary model)
            attention = None
            for model_id in ["platinum_sensitivity", "tumor_grade", "survival_5y"]:
                if model_id in state.predictions and "attention" in state.predictions[model_id]:
                    attention = np.array(state.predictions[model_id]["attention"])
                    break
            
            if attention is not None:
                top_k = min(8, len(attention))
                top_indices = np.argsort(attention)[-top_k:][::-1]
                
                for i, idx in enumerate(top_indices):
                    evidence = {
                        "rank": i + 1,
                        "patch_index": int(idx),
                        "attention_weight": float(attention[idx]),
                    }
                    if state.context.coordinates is not None and idx < len(state.context.coordinates):
                        evidence["coordinates"] = [
                            int(state.context.coordinates[idx][0]),
                            int(state.context.coordinates[idx][1]),
                        ]
                    state.top_evidence.append(evidence)
            
            # Build reasoning about predictions
            reasoning_parts = ["Model predictions:"]
            
            for model_id, pred in state.predictions.items():
                if "error" not in pred:
                    label = pred.get("label", "unknown")
                    score = pred.get("score", 0)
                    confidence = pred.get("confidence", 0)
                    reasoning_parts.append(
                        f"- {pred.get('model_name', model_id)}: {label} "
                        f"(score: {score:.2f}, confidence: {confidence:.1%})"
                    )
            
            state.reasoning_chain.append("\n".join(reasoning_parts))
            
            duration = (time.time() - start) * 1000
            
            result = StepResult(
                step=AgentStep.ANALYZE,
                status=StepStatus.COMPLETE,
                message=f"Ran {len(state.predictions)} models on {state.context.n_patches:,} patches",
                data={
                    "predictions": {
                        k: {key: v[key] for key in ["model_name", "label", "score", "confidence"] if key in v}
                        for k, v in state.predictions.items()
                        if "error" not in v
                    },
                    "top_evidence": state.top_evidence[:5],
                },
                reasoning="\n".join(reasoning_parts),
                duration_ms=duration,
            )
            state.step_results[AgentStep.ANALYZE] = result
            return result
            
        except Exception as e:
            logger.error(f"Analysis step failed: {e}")
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.ANALYZE,
                status=StepStatus.ERROR,
                message=f"Analysis failed: {str(e)}",
                duration_ms=duration,
            )
            state.step_results[AgentStep.ANALYZE] = result
            return result
    
    async def _step_retrieve(self, state: AgentState) -> StepResult:
        """Retrieve: Find similar cases from reference database."""
        import time
        start = time.time()
        
        state.current_step = AgentStep.RETRIEVE
        
        if self.slide_mean_index is None:
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.RETRIEVE,
                status=StepStatus.SKIPPED,
                message="Similarity index not available",
                reasoning="No FAISS index built. Similar case retrieval skipped.",
                duration_ms=duration,
            )
            state.step_results[AgentStep.RETRIEVE] = result
            return result
        
        try:
            # Compute mean embedding for slide
            q = state.context.embeddings.astype(np.float32).mean(axis=0)
            q = q / (np.linalg.norm(q) + 1e-12)
            q = q.reshape(1, -1).astype(np.float32)
            
            # Search FAISS index
            k = 10
            search_k = min(len(self.slide_mean_ids), max(k + 10, k * 3))
            sims, idxs = self.slide_mean_index.search(q, search_k)
            
            slide_id = state.context.slide_id
            seen = set()
            
            for sim, idx in zip(sims[0], idxs[0]):
                if idx < 0 or idx >= len(self.slide_mean_ids):
                    continue
                sid = self.slide_mean_ids[int(idx)]
                if sid == slide_id or sid in seen:
                    continue
                seen.add(sid)
                
                meta = self.slide_mean_meta.get(sid, {})
                label = meta.get("label") or self.slide_labels.get(sid)
                
                state.similar_cases.append({
                    "slide_id": sid,
                    "similarity_score": float(sim),
                    "label": label,
                    "n_patches": meta.get("n_patches"),
                })
                
                if len(state.similar_cases) >= k:
                    break
            
            # Build reasoning
            responders = sum(1 for c in state.similar_cases if c.get("label") == "responder")
            non_responders = sum(1 for c in state.similar_cases if c.get("label") == "non-responder")
            unknown = len(state.similar_cases) - responders - non_responders
            
            reasoning = (
                f"Found {len(state.similar_cases)} similar cases based on morphological embedding similarity.\n"
                f"Among similar cases: {responders} responders, {non_responders} non-responders"
            )
            if unknown > 0:
                reasoning += f", {unknown} unknown"
            
            state.reasoning_chain.append(reasoning)
            
            duration = (time.time() - start) * 1000
            
            result = StepResult(
                step=AgentStep.RETRIEVE,
                status=StepStatus.COMPLETE,
                message=f"Found {len(state.similar_cases)} similar cases",
                data={"similar_cases": state.similar_cases[:5]},
                reasoning=reasoning,
                duration_ms=duration,
            )
            state.step_results[AgentStep.RETRIEVE] = result
            return result
            
        except Exception as e:
            logger.error(f"Retrieval step failed: {e}")
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.RETRIEVE,
                status=StepStatus.ERROR,
                message=f"Retrieval failed: {str(e)}",
                duration_ms=duration,
            )
            state.step_results[AgentStep.RETRIEVE] = result
            return result
    
    async def _step_semantic_search(self, state: AgentState) -> StepResult:
        """Semantic Search: Run key pathology queries via MedSigLIP to find tissue patterns."""
        import time
        start = time.time()

        state.current_step = AgentStep.SEMANTIC_SEARCH

        if self.medsiglip_embedder is None:
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.SEMANTIC_SEARCH,
                status=StepStatus.SKIPPED,
                message="MedSigLIP embedder not available",
                reasoning="Semantic tissue search skipped — MedSigLIP model not loaded.",
                duration_ms=duration,
            )
            state.step_results[AgentStep.SEMANTIC_SEARCH] = result
            return result

        try:
            slide_id = state.context.slide_id

            # Load or compute MedSigLIP embeddings for this slide
            siglip_cache_path = self.embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip.npy"
            if siglip_cache_path.exists():
                siglip_embeddings = np.load(siglip_cache_path)
            else:
                # Fall back: cannot compute on-the-fly without WSI access in the workflow
                duration = (time.time() - start) * 1000
                result = StepResult(
                    step=AgentStep.SEMANTIC_SEARCH,
                    status=StepStatus.SKIPPED,
                    message="MedSigLIP patch embeddings not cached for this slide",
                    reasoning="Pre-computed MedSigLIP embeddings not found. Run semantic embedding first.",
                    duration_ms=duration,
                )
                state.step_results[AgentStep.SEMANTIC_SEARCH] = result
                return result

            # Build metadata for each patch
            metadata = []
            for i in range(len(siglip_embeddings)):
                meta: Dict[str, Any] = {"index": i}
                if state.context.coordinates is not None and i < len(state.context.coordinates):
                    meta["coordinates"] = [
                        int(state.context.coordinates[i][0]),
                        int(state.context.coordinates[i][1]),
                    ]
                metadata.append(meta)

            # Key pathology queries to characterise the tissue
            queries = [
                "tumor cells and malignant tissue",
                "necrotic tissue and cell death",
                "lymphocyte infiltration and immune cells",
                "stromal and connective tissue",
                "mitotic figures and cell division",
            ]

            all_results: Dict[str, list] = {}
            reasoning_parts = ["Semantic tissue search results (MedSigLIP):"]

            for query in queries:
                hits = self.medsiglip_embedder.search(
                    query=query,
                    embeddings=siglip_embeddings,
                    metadata=metadata,
                    top_k=5,
                )
                short_name = query.split(" and ")[0].strip()
                all_results[short_name] = hits

                if hits:
                    best = hits[0]
                    reasoning_parts.append(
                        f"- {short_name}: top match score {best['similarity_score']:.3f} "
                        f"(patch #{best['patch_index']})"
                    )

            state.semantic_search_results = all_results
            reasoning = "\n".join(reasoning_parts)
            state.reasoning_chain.append(reasoning)

            duration = (time.time() - start) * 1000

            result = StepResult(
                step=AgentStep.SEMANTIC_SEARCH,
                status=StepStatus.COMPLETE,
                message=f"Searched {len(queries)} pathology patterns across {len(siglip_embeddings)} patches",
                data={"semantic_search": {k: v[:3] for k, v in all_results.items()}},
                reasoning=reasoning,
                duration_ms=duration,
            )
            state.step_results[AgentStep.SEMANTIC_SEARCH] = result
            return result

        except Exception as e:
            logger.error(f"Semantic search step failed: {e}")
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.SEMANTIC_SEARCH,
                status=StepStatus.ERROR,
                message=f"Semantic search failed: {str(e)}",
                duration_ms=duration,
            )
            state.step_results[AgentStep.SEMANTIC_SEARCH] = result
            return result

    async def _step_compare(self, state: AgentState) -> StepResult:
        """Compare: Analyze similarities and differences with reference cases."""
        import time
        start = time.time()
        
        state.current_step = AgentStep.COMPARE
        
        if not state.similar_cases:
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.COMPARE,
                status=StepStatus.SKIPPED,
                message="No similar cases to compare",
                reasoning="Skipped comparison step due to no similar cases found.",
                duration_ms=duration,
            )
            state.step_results[AgentStep.COMPARE] = result
            return result
        
        # Analyze patterns in similar cases
        responders = [c for c in state.similar_cases if c.get("label") == "responder"]
        non_responders = [c for c in state.similar_cases if c.get("label") == "non-responder"]
        
        comparison_insights = []
        
        # Get primary prediction
        primary_pred = None
        for model_id in ["platinum_sensitivity", "tumor_grade"]:
            if model_id in state.predictions:
                primary_pred = state.predictions[model_id]
                break
        
        if primary_pred and responders:
            most_similar_resp = responders[0]
            comparison_insights.append(
                f"Most similar responding case: {most_similar_resp['slide_id']} "
                f"(similarity: {most_similar_resp['similarity_score']:.2%})"
            )
        
        if primary_pred and non_responders:
            most_similar_nonresp = non_responders[0]
            comparison_insights.append(
                f"Most similar non-responding case: {most_similar_nonresp['slide_id']} "
                f"(similarity: {most_similar_nonresp['similarity_score']:.2%})"
            )
        
        # Compute weighted vote from similar cases
        if responders or non_responders:
            resp_weight = sum(c["similarity_score"] for c in responders)
            nonresp_weight = sum(c["similarity_score"] for c in non_responders)
            total_weight = resp_weight + nonresp_weight
            
            if total_weight > 0:
                resp_pct = resp_weight / total_weight * 100
                comparison_insights.append(
                    f"Similarity-weighted case vote: {resp_pct:.0f}% responder, {100-resp_pct:.0f}% non-responder"
                )
        
        reasoning = "Comparison with similar cases:\n" + "\n".join(f"- {i}" for i in comparison_insights)
        state.reasoning_chain.append(reasoning)
        
        duration = (time.time() - start) * 1000
        
        result = StepResult(
            step=AgentStep.COMPARE,
            status=StepStatus.COMPLETE,
            message=f"Compared with {len(responders)} responders and {len(non_responders)} non-responders",
            data={
                "comparison_insights": comparison_insights,
                "responder_count": len(responders),
                "non_responder_count": len(non_responders),
            },
            reasoning=reasoning,
            duration_ms=duration,
        )
        state.step_results[AgentStep.COMPARE] = result
        return result
    
    async def _step_reason(self, state: AgentState) -> StepResult:
        """Reason: Generate evidence-based reasoning (optionally with MedGemma)."""
        import time
        start = time.time()
        
        state.current_step = AgentStep.REASON
        
        # Build reasoning from collected evidence
        reasoning_parts = ["Evidence-Based Analysis:"]
        
        # Add model prediction context
        if state.predictions:
            for model_id, pred in state.predictions.items():
                if "error" not in pred:
                    confidence = pred.get("confidence", 0)
                    level = "high" if confidence > 0.6 else "moderate" if confidence > 0.3 else "low"
                    reasoning_parts.append(
                        f"\n{pred.get('model_name', model_id)} ({level} confidence):\n"
                        f"  Prediction: {pred.get('label')} (score: {pred.get('score', 0):.3f})"
                    )
        
        # Add evidence patch context
        if state.top_evidence:
            reasoning_parts.append(f"\nTop {len(state.top_evidence)} high-attention regions identified:")
            for ev in state.top_evidence[:3]:
                coords = ev.get("coordinates", [0, 0])
                reasoning_parts.append(
                    f"  - Region at ({coords[0]:,}, {coords[1]:,}): "
                    f"attention weight {ev['attention_weight']:.3f}"
                )
        
        # Add similar case context
        if state.similar_cases:
            responders = sum(1 for c in state.similar_cases if c.get("label") == "responder")
            non_responders = sum(1 for c in state.similar_cases if c.get("label") == "non-responder")
            reasoning_parts.append(
                f"\nSimilar case analysis: {responders}/{len(state.similar_cases)} similar cases were responders"
            )
        
        # Add semantic search context
        if state.semantic_search_results:
            reasoning_parts.append("\nSemantic tissue pattern analysis (MedSigLIP):")
            for query_name, hits in state.semantic_search_results.items():
                if hits:
                    best = hits[0]
                    score = best.get("similarity_score", 0)
                    strength = "strong" if score > 0.3 else "moderate" if score > 0.2 else "weak"
                    reasoning_parts.append(
                        f"  - {query_name}: {strength} signal (score: {score:.3f})"
                    )
        
        # Answer user questions if provided
        if state.context.questions:
            reasoning_parts.append("\nAnswering your questions:")
            for q in state.context.questions:
                answer = self._generate_answer(q, state)
                reasoning_parts.append(f"\nQ: {q}\nA: {answer}")
        
        full_reasoning = "\n".join(reasoning_parts)
        state.reasoning_chain.append(full_reasoning)
        
        duration = (time.time() - start) * 1000
        
        result = StepResult(
            step=AgentStep.REASON,
            status=StepStatus.COMPLETE,
            message="Generated evidence-based reasoning",
            data={"reasoning": full_reasoning},
            reasoning=full_reasoning,
            duration_ms=duration,
        )
        state.step_results[AgentStep.REASON] = result
        return result

    async def _step_report(self, state: AgentState) -> StepResult:
        """Report: Generate final structured report (with MedGemma when available)."""
        import time
        start = time.time()

        state.current_step = AgentStep.REPORT

        # Determine primary prediction to report.
        label = "unknown"
        score = 0.0
        if state.predictions:
            # Prefer the project's primary model, then common treatment keys
            preferred_keys = [
                "treatment_response",
                "bevacizumab_response",
                "platinum_sensitivity",
            ]
            pred = None
            for key in preferred_keys:
                if key in state.predictions and "error" not in state.predictions[key]:
                    pred = state.predictions[key]
                    break
            if pred is None:
                for v in state.predictions.values():
                    if "error" not in v:
                        pred = v
                        break
            if pred is not None:
                label = str(pred.get("label", label))
                score = float(pred.get("score", pred.get("probability", score)) or score)

        try:
            if self.medgemma_reporter is not None:
                # Use MedGemma for structured clinical report
                evidence_patches = []
                for ev in state.top_evidence[:8]:
                    coords = ev.get("coordinates") or [0, 0]
                    evidence_patches.append({
                        "patch_id": f"patch_{ev.get('rank', 0)}",
                        "coordinates": coords,
                        "attention_weight": ev.get("attention_weight"),
                        "morphology_description": "high-attention region",
                        "significance": "High attention region",
                    })

                # Run MedGemma inference in a thread to avoid blocking the
                # async event loop (CPU inference takes ~120-180s).
                import asyncio
                import functools

                _gen = functools.partial(
                    self.medgemma_reporter.generate_report,
                    evidence_patches=evidence_patches,
                    score=score,
                    label=label,
                    similar_cases=state.similar_cases,
                    case_id=state.context.slide_id,
                    patient_context=None,
                )
                state.report = await asyncio.to_thread(_gen)
                status = StepStatus.COMPLETE
                message = "Generated structured clinical report via MedGemma"
                reasoning = "Generated a structured report grounded in model outputs and high-attention evidence regions."
            else:
                # Build a rich structured report from collected evidence (no MedGemma)
                report: Dict[str, Any] = {
                    "case_id": state.context.slide_id,
                    "task": "Multi-model slide analysis for treatment response prediction",
                    "clinical_context": state.context.clinical_context,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "predictions": {},
                    "evidence": [],
                    "similar_cases": [],
                    "semantic_search": {},
                    "reasoning_summary": "",
                    "limitations": [
                        "This is a research model and should not be used for clinical decisions without validation",
                        "Predictions are based on morphological features only",
                        "Model confidence should be considered when interpreting results",
                    ],
                    "suggested_next_steps": [
                        "Review evidence patches and prediction outputs",
                        "Correlate with clinical context",
                        "Consider molecular testing for confirmation",
                    ],
                    "safety_statement": (
                        "This analysis is for research and decision support only. "
                        "All findings must be validated by qualified pathologists and clinicians."
                    ),
                }

                for model_id, p in state.predictions.items():
                    if "error" not in p:
                        report["predictions"][model_id] = {
                            "model_name": p.get("model_name", model_id),
                            "label": p.get("label"),
                            "score": p.get("score"),
                            "confidence": p.get("confidence"),
                        }

                for ev in state.top_evidence[:5]:
                    report["evidence"].append({
                        "rank": ev["rank"],
                        "patch_index": ev["patch_index"],
                        "attention_weight": ev["attention_weight"],
                        "coordinates": ev.get("coordinates"),
                    })

                for case in state.similar_cases[:5]:
                    report["similar_cases"].append({
                        "slide_id": case["slide_id"],
                        "similarity_score": case["similarity_score"],
                        "label": case.get("label"),
                    })

                # Include semantic search highlights
                for query_name, hits in state.semantic_search_results.items():
                    report["semantic_search"][query_name] = [
                        {
                            "patch_index": h["patch_index"],
                            "similarity_score": h["similarity_score"],
                            "coordinates": h.get("metadata", {}).get("coordinates"),
                        }
                        for h in hits[:3]
                    ]

                report["reasoning_summary"] = "\n\n".join(state.reasoning_chain)
                state.report = report
                status = StepStatus.COMPLETE
                message = "Generated structured analysis report"
                reasoning = (
                    f"Report generated with {len(report['predictions'])} model predictions, "
                    f"{len(report['evidence'])} evidence regions, "
                    f"{len(report['similar_cases'])} similar cases, "
                    f"and {len(report['semantic_search'])} semantic queries."
                )

            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.REPORT,
                status=status,
                message=message,
                data={"report": state.report},
                reasoning=reasoning,
                duration_ms=duration,
            )
            state.step_results[AgentStep.REPORT] = result
            return result

        except Exception as e:
            logger.error(f"Report step failed: {e}")
            duration = (time.time() - start) * 1000
            result = StepResult(
                step=AgentStep.REPORT,
                status=StepStatus.ERROR,
                message=f"Report generation failed: {str(e)}",
                reasoning="Report generation encountered an exception.",
                duration_ms=duration,
            )
            state.step_results[AgentStep.REPORT] = result
            return result

    def _generate_answer(self, question: str, state: AgentState) -> str:
        """Generate a context-aware answer to a user question based on analysis results."""
        q_lower = question.lower()
        
        # Check for treatment response questions
        if any(word in q_lower for word in ["response", "treatment", "platinum", "sensitive", "resistant", "chemo"]):
            # Find the primary treatment response model (project-aware)
            treatment_keys = ["platinum_sensitivity", "treatment_response", "bevacizumab_response"]
            pred = None
            pred_model_name = "treatment response"
            for key in treatment_keys:
                if key in state.predictions and "error" not in state.predictions[key]:
                    pred = state.predictions[key]
                    pred_model_name = pred.get("model_name", key.replace("_", " ").title())
                    break
            # Fallback: use first non-error prediction
            if pred is None:
                for v in state.predictions.values():
                    if "error" not in v:
                        pred = v
                        pred_model_name = pred.get("model_name", "treatment response")
                        break
            
            if pred is not None:
                score = pred.get("score", 0.5)
                label = pred.get("label", "unknown")
                confidence = pred.get("confidence", 0)
                
                response = f"Based on the {pred_model_name} model, the predicted response is **{label.upper()}** "
                response += f"with a probability of {score:.1%}.\n\n"
                
                if confidence > 0.6:
                    response += "The model shows **high confidence** in this prediction.\n"
                elif confidence > 0.3:
                    response += "The model shows **moderate confidence**; pathologist review is recommended.\n"
                else:
                    response += "The model is **uncertain**; additional testing may be warranted.\n"
                
                # Add context from similar cases
                if state.similar_cases:
                    resp_count = sum(1 for c in state.similar_cases if c.get("label") == "responder")
                    non_resp = len(state.similar_cases) - resp_count
                    response += f"\nAmong {len(state.similar_cases)} morphologically similar cases:\n"
                    response += f"• {resp_count} were responders\n"
                    response += f"• {non_resp} were non-responders\n"
                    
                    if resp_count > non_resp and "responder" in label.lower():
                        response += "\nThis aligns with the model prediction."
                    elif resp_count < non_resp and "non" in label.lower():
                        response += "\nThis aligns with the model prediction."
                    else:
                        response += "\nNote: Similar cases show mixed outcomes."
                
                return response
            return "Treatment response model not available for this analysis."
        
        # Check for why/reasoning questions
        if any(word in q_lower for word in ["why", "reason", "explain", "how did"]):
            response = "The prediction is based on several factors:\n\n"
            
            # Model predictions
            if state.predictions:
                response += "**Model Analysis:**\n"
                for model_id, pred in state.predictions.items():
                    if "error" not in pred:
                        response += f"• {pred.get('model_name', model_id)}: {pred.get('label')} ({pred.get('score', 0):.1%})\n"
                response += "\n"
            
            # Evidence regions
            if state.top_evidence:
                response += f"**Key Evidence Regions:**\n"
                response += f"The model identified {len(state.top_evidence)} high-attention tissue regions that drove this prediction. "
                response += "These areas showed morphological patterns most associated with the predicted outcome.\n\n"
            
            # Similar cases
            if state.similar_cases:
                resp_count = sum(1 for c in state.similar_cases if c.get("label") == "responder")
                response += f"**Similar Cases:**\n"
                response += f"Among {len(state.similar_cases)} morphologically similar slides, {resp_count} were responders. "
                response += "This historical context supports the prediction confidence level.\n"
            
            return response
        
        # Check for survival questions
        if any(word in q_lower for word in ["survival", "prognosis", "outcome", "year", "mortality"]):
            survival_models = ["survival_5y", "survival_3y", "survival_1y"]
            responses = []
            for model_id in survival_models:
                if model_id in state.predictions:
                    pred = state.predictions[model_id]
                    label = pred.get("label", "unknown")
                    score = pred.get("score", 0)
                    years = model_id.split("_")[1]
                    
                    if label == "positive" or score > 0.5:
                        outcome = "favorable"
                    else:
                        outcome = "concerning"
                    
                    responses.append(f"• **{years} survival**: {outcome} ({score:.1%} positive probability)")
            
            if responses:
                return "**Survival Predictions:**\n\n" + "\n".join(responses) + \
                    "\n\nThese predictions should be interpreted alongside clinical factors including stage, treatment history, and molecular markers."
            return "Survival models not available for this analysis."
        
        # Check for region/heatmap questions
        if any(word in q_lower for word in ["region", "area", "attention", "heatmap", "concerning", "patch", "evidence", "show"]):
            if state.top_evidence:
                response = f"The model identified **{len(state.top_evidence)} high-attention regions**:\n\n"
                for ev in state.top_evidence[:5]:
                    coords = ev.get("coordinates", [0, 0])
                    weight = ev["attention_weight"]
                    intensity = "High" if weight > 0.05 else "Moderate" if weight > 0.02 else "Low"
                    response += f"• **Region #{ev['rank']}** at ({coords[0]:,}, {coords[1]:,}) — {intensity} attention ({weight:.3f})\n"
                response += "\nClick on the region buttons to navigate to these areas in the slide viewer."
                response += "\n\nThese regions contributed most to the model's prediction and should be reviewed by the pathologist."
                return response
            return "No high-attention regions identified in this analysis."
        
        # Check for comparison questions
        if any(word in q_lower for word in ["compare", "similar", "cases", "other", "like this"]):
            if state.similar_cases:
                response = f"**Comparison with {len(state.similar_cases)} Similar Cases:**\n\n"
                
                responders = [c for c in state.similar_cases if c.get("label") == "responder"]
                non_responders = [c for c in state.similar_cases if c.get("label") == "non-responder"]
                
                response += f"• {len(responders)} responders ({len(responders)/len(state.similar_cases)*100:.0f}%)\n"
                response += f"• {len(non_responders)} non-responders ({len(non_responders)/len(state.similar_cases)*100:.0f}%)\n\n"
                
                response += "**Top Matches:**\n"
                for c in state.similar_cases[:3]:
                    sim = c["similarity_score"]
                    label = c.get("label", "unknown")
                    response += f"• `{c['slide_id']}` — {sim*100:.0f}% match ({label})\n"
                
                return response
            return "No similar cases available for comparison."
        
        # Check for alternative treatment questions
        if any(word in q_lower for word in ["alternative", "other treatment", "instead", "option"]):
            # Find any treatment response prediction
            treatment_pred = None
            for key in ["platinum_sensitivity", "treatment_response", "bevacizumab_response"]:
                if key in state.predictions and "error" not in state.predictions[key]:
                    treatment_pred = state.predictions[key]
                    break
            if treatment_pred is None:
                for v in state.predictions.values():
                    if "error" not in v:
                        treatment_pred = v
                        break
            
            if treatment_pred is not None:
                label = treatment_pred.get("label", "unknown").lower()
                is_negative = "non" in label or "resistant" in label or "negative" in label or "poor" in label
                
                if is_negative:
                    return """For predicted treatment-resistant cases, alternative strategies may include:

**Consider:**
• Targeted therapy based on molecular profiling results
• Immunotherapy if eligible based on biomarker status
• Clinical trial enrollment
• Alternative chemotherapy regimens

Treatment decisions should be made by the multidisciplinary tumor board considering:
• Molecular profiling results
• Prior treatment history
• Performance status
• Patient preferences

⚠️ This is a research tool — consult with the clinical team for treatment planning."""
                else:
                    return """For predicted treatment-responsive cases, standard therapy may be appropriate:

**Consider:**
• Standard-of-care chemotherapy regimen for this cancer type
• Maintenance therapy based on molecular profile
• Regular monitoring for treatment response

Treatment decisions should incorporate:
• Molecular profiling results
• Clinical staging
• Patient factors and preferences

The prediction suggests favorable treatment response based on morphological patterns.

⚠️ This is a research tool — consult with the clinical team for treatment planning."""
            return "Run analysis first to get treatment recommendations."
        
        # Check for confidence questions
        if any(word in q_lower for word in ["confident", "certain", "sure", "reliable", "trust"]):
            if state.predictions:
                response = "**Model Confidence Assessment:**\n\n"
                for model_id, pred in state.predictions.items():
                    if "error" not in pred:
                        conf = pred.get("confidence", 0)
                        if conf > 0.6:
                            level = "HIGH"
                        elif conf > 0.3:
                            level = "MODERATE"
                        else:
                            level = "LOW"
                        response += f"• {pred.get('model_name', model_id)}: {level} ({conf:.1%})\n"
                
                response += "\n**Interpretation:**\n"
                avg_conf = sum(p.get("confidence", 0) for p in state.predictions.values() if "error" not in p) / max(1, len(state.predictions))
                if avg_conf > 0.6:
                    response += "Overall high confidence. Predictions are reliable for clinical decision support."
                elif avg_conf > 0.3:
                    response += "Moderate confidence. Consider additional testing or expert review."
                else:
                    response += "Low confidence. Recommend additional molecular testing and pathology review."
                
                return response
            return "Run analysis first to assess model confidence."
        
        # Check for differential diagnosis
        if any(word in q_lower for word in ["differential", "diagnos", "possibilities"]):
            response = "**Differential Considerations Based on Analysis:**\n\n"
            
            # Find any treatment prediction for context
            treatment_pred = None
            for key in ["platinum_sensitivity", "treatment_response", "bevacizumab_response"]:
                if key in state.predictions and "error" not in state.predictions[key]:
                    treatment_pred = state.predictions[key]
                    break
            if treatment_pred is None:
                for v in state.predictions.values():
                    if "error" not in v:
                        treatment_pred = v
                        break
            
            if treatment_pred is not None:
                label = treatment_pred.get("label", "unknown").lower()
                is_negative = "non" in label or "resistant" in label or "negative" in label or "poor" in label
                if is_negative:
                    response += "The morphological patterns suggest potential treatment resistance. Consider:\n\n"
                    response += "• **Intrinsic resistance**: Pre-existing molecular alterations\n"
                    response += "• **High tumor heterogeneity**: Mixed cell populations\n"
                    response += "• **Mesenchymal features**: EMT-related patterns\n"
                else:
                    response += "The morphological patterns suggest treatment sensitivity. Features include:\n\n"
                    response += "• **Homogeneous tumor architecture**: Consistent cellular patterns\n"
                    response += "• **Typical cancer morphology**: Characteristic histological features\n"
            
            if state.similar_cases:
                response += f"\n**Historical Context:**\n"
                response += f"Similar cases ({len(state.similar_cases)}) showed diverse outcomes, "
                response += "emphasizing the importance of molecular confirmation.\n"
            
            response += "\nThis AI analysis is for decision support only. Final diagnosis requires expert pathology review."
            return response
        
        # Default response with helpful suggestions
        return """I can help you understand this analysis. Try asking about:

• **Treatment response**: "What is the predicted treatment response?"
• **Reasoning**: "Why was this prediction made?"
• **Evidence regions**: "Show me the high-attention areas"
• **Similar cases**: "How does this compare to similar cases?"
• **Survival**: "What is the survival prognosis?"
• **Confidence**: "How confident is the model?"
• **Alternatives**: "What are the treatment alternatives?"
• **Differential**: "What are the differential diagnoses?"

Please rephrase your question using one of these topics."""

    async def followup(
        self,
        session_id: str,
        question: str,
    ) -> AsyncIterator[StepResult]:
        """
        Handle a follow-up question for an existing session.
        
        Uses the stored analysis context to answer questions about:
        - Treatment response predictions
        - Evidence regions
        - Similar cases
        - Model confidence
        - Prognosis
        """
        state = self._sessions.get(session_id)
        if state is None:
            yield StepResult(
                step=AgentStep.ERROR,
                status=StepStatus.ERROR,
                message=f"Session {session_id} not found",
            )
            return
        
        # Add to conversation history
        state.conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        
        # Generate answer
        answer = self._generate_answer(question, state)
        
        # Add response to history
        state.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        
        yield StepResult(
            step=AgentStep.REASON,
            status=StepStatus.COMPLETE,
            message="Generated response",
            reasoning=answer,
            data={
                "question": question,
                "session_id": session_id,
            },
        )
    
    def get_result(self, session_id: str) -> Optional[AgentResult]:
        """Get the final result for a completed session."""
        state = self._sessions.get(session_id)
        if state is None:
            return None
        
        total_duration = sum(
            r.duration_ms for r in state.step_results.values()
        )
        
        return AgentResult(
            session_id=state.session_id,
            slide_id=state.context.slide_id,
            success=state.error is None,
            predictions={
                k: {key: v[key] for key in ["model_name", "label", "score", "confidence"] if key in v}
                for k, v in state.predictions.items()
                if "error" not in v
            },
            similar_cases=state.similar_cases,
            top_evidence=state.top_evidence,
            semantic_search_results=state.semantic_search_results,
            report=state.report,
            reasoning_chain=state.reasoning_chain,
            total_duration_ms=total_duration,
            error=state.error,
        )
