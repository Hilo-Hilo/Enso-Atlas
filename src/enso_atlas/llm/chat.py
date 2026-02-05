"""
RAG-based Chat Interface for Enso Atlas.

Provides conversational AI with context from:
- Clinical reports (M5)
- Slide analysis results
- Similar cases from the knowledge base

Implements a simple RAG pattern:
1. Retrieve relevant context (report, predictions, similar cases)
2. Format context into prompt
3. Generate response using rule-based reasoning or LLM
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator
import logging
import uuid
import re

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    evidence_patches: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatContext:
    """Context retrieved for RAG."""
    slide_id: str
    report: Optional[Dict[str, Any]] = None
    predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    top_evidence: List[Dict[str, Any]] = field(default_factory=list)
    clinical_context: str = ""


@dataclass
class ChatSession:
    """
    Manages a chat session with context and history.
    
    Maintains:
    - Chat history for multi-turn conversation
    - Retrieved context from slide analysis
    - Reference to generated report
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    slide_id: Optional[str] = None
    history: List[Message] = field(default_factory=list)
    context: Optional[ChatContext] = None
    
    def add_message(self, role: str, content: str, evidence_patches: Optional[List] = None) -> Message:
        """Add a message to the history."""
        msg = Message(role=role, content=content, evidence_patches=evidence_patches)
        self.history.append(msg)
        return msg
    
    def get_history_text(self, max_messages: int = 10) -> str:
        """Format recent history for prompt context."""
        recent = self.history[-max_messages:]
        lines = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)


class ChatManager:
    """
    Manages chat sessions and generates responses using RAG.
    
    Uses retrieved context from:
    - Generated clinical reports
    - Model predictions
    - Similar cases
    - Evidence patches
    """
    
    def __init__(
        self,
        embeddings_dir=None,
        multi_model_inference=None,
        evidence_generator=None,
        medgemma_reporter=None,
        slide_labels=None,
        slide_mean_index=None,
        slide_mean_ids=None,
        slide_mean_meta=None,
    ):
        self.embeddings_dir = embeddings_dir
        self.multi_model_inference = multi_model_inference
        self.evidence_generator = evidence_generator
        self.medgemma_reporter = medgemma_reporter
        self.slide_labels = slide_labels or {}
        self.slide_mean_index = slide_mean_index
        self.slide_mean_ids = slide_mean_ids or []
        self.slide_mean_meta = slide_mean_meta or {}
        
        # In-memory session storage
        self._sessions: Dict[str, ChatSession] = {}
        
        # Cache for slide analysis results
        self._analysis_cache: Dict[str, ChatContext] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None, slide_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            # Update slide_id if provided
            if slide_id and session.slide_id != slide_id:
                session.slide_id = slide_id
                session.context = self._analysis_cache.get(slide_id)
            return session
        
        # Create new session
        session = ChatSession(slide_id=slide_id)
        if slide_id:
            session.context = self._analysis_cache.get(slide_id)
        self._sessions[session.session_id] = session
        logger.info(f"Created chat session {session.session_id} for slide {slide_id}")
        return session
    
    async def retrieve_context(self, slide_id: str) -> ChatContext:
        """
        Retrieve context for RAG from slide analysis.
        
        Loads:
        - Pre-computed report if available
        - Model predictions
        - Similar cases
        - Evidence patches
        """
        import numpy as np
        
        # Check cache first
        if slide_id in self._analysis_cache:
            return self._analysis_cache[slide_id]
        
        context = ChatContext(slide_id=slide_id)
        
        if self.embeddings_dir is None:
            return context
        
        emb_path = self.embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            logger.warning(f"Embeddings not found for {slide_id}")
            return context
        
        # Load embeddings
        embeddings = np.load(emb_path)
        
        # Run multi-model inference if available
        if self.multi_model_inference is not None:
            try:
                results = self.multi_model_inference.predict_all(
                    embeddings,
                    return_attention=True,
                )
                context.predictions = results.get("predictions", {})
                
                # Extract top evidence
                for model_id in ["platinum_sensitivity", "tumor_grade", "survival_5y"]:
                    if model_id in context.predictions and "attention" in context.predictions[model_id]:
                        attention = np.array(context.predictions[model_id]["attention"])
                        top_k = min(8, len(attention))
                        top_indices = np.argsort(attention)[-top_k:][::-1]
                        
                        # Load coordinates
                        coord_path = self.embeddings_dir / f"{slide_id}_coords.npy"
                        coords = np.load(coord_path) if coord_path.exists() else None
                        
                        for i, idx in enumerate(top_indices):
                            evidence = {
                                "rank": i + 1,
                                "patch_index": int(idx),
                                "attention_weight": float(attention[idx]),
                            }
                            if coords is not None and idx < len(coords):
                                evidence["coordinates"] = [int(coords[idx][0]), int(coords[idx][1])]
                            context.top_evidence.append(evidence)
                        break
                        
            except Exception as e:
                logger.warning(f"Multi-model inference failed: {e}")
        
        # Find similar cases
        if self.slide_mean_index is not None:
            try:
                q = embeddings.astype(np.float32).mean(axis=0)
                q = q / (np.linalg.norm(q) + 1e-12)
                q = q.reshape(1, -1).astype(np.float32)
                
                k = 10
                search_k = min(len(self.slide_mean_ids), max(k + 10, k * 3))
                sims, idxs = self.slide_mean_index.search(q, search_k)
                
                seen = set()
                for sim, idx in zip(sims[0], idxs[0]):
                    if idx < 0 or idx >= len(self.slide_mean_ids):
                        continue
                    sid = self.slide_mean_ids[int(idx)]
                    if sid == slide_id or sid in seen:
                        continue
                    seen.add(sid)
                    
                    meta = self.slide_mean_meta.get(sid, {})
                    context.similar_cases.append({
                        "slide_id": sid,
                        "similarity_score": float(sim),
                        "label": meta.get("label") or self.slide_labels.get(sid),
                    })
                    if len(context.similar_cases) >= k:
                        break
                        
            except Exception as e:
                logger.warning(f"Similar case search failed: {e}")
        
        # Cache the context
        self._analysis_cache[slide_id] = context
        return context
    
    def _generate_response(self, question: str, context: ChatContext, history: List[Message]) -> str:
        """
        Generate a response using retrieved context.
        
        Uses rule-based reasoning with context augmentation.
        For production, this would call MedGemma or another LLM.
        """
        q_lower = question.lower()
        
        # Get primary prediction for context
        primary_pred = None
        for model_id in ["platinum_sensitivity", "tumor_grade", "survival_5y"]:
            if model_id in context.predictions and "error" not in context.predictions.get(model_id, {}):
                primary_pred = context.predictions[model_id]
                break
        
        # === PROGNOSIS / SURVIVAL QUESTIONS ===
        if any(word in q_lower for word in ["prognosis", "survival", "outcome", "life expectancy", "how long"]):
            survival_models = ["survival_5y", "survival_3y", "survival_1y"]
            responses = []
            
            for model_id in survival_models:
                if model_id in context.predictions and "error" not in context.predictions.get(model_id, {}):
                    pred = context.predictions[model_id]
                    label = pred.get("label", "unknown")
                    score = pred.get("score", 0)
                    years = model_id.split("_")[1]
                    
                    if label == "positive" or score > 0.5:
                        emoji = "🟢"
                        outcome = "favorable"
                    else:
                        emoji = "🔴"
                        outcome = "concerning"
                    
                    responses.append(f"• **{years} survival**: {emoji} {outcome} ({score:.1%} positive probability)")
            
            if responses:
                result = "**Survival Predictions Based on Morphological Analysis:**\n\n"
                result += "\n".join(responses)
                
                # Add context from similar cases
                if context.similar_cases:
                    resp_count = sum(1 for c in context.similar_cases if c.get("label") == "responder")
                    result += f"\n\n**Similar Case Context:**\n"
                    result += f"Among {len(context.similar_cases)} morphologically similar cases, "
                    result += f"{resp_count} showed positive treatment response.\n"
                
                result += "\n\n⚠️ **Important**: These predictions are based on morphological features only. "
                result += "Final prognosis should consider stage, molecular markers (BRCA/HRD status), "
                result += "performance status, and treatment response."
                return result
            
            # Fallback if no survival models
            if primary_pred:
                return f"""**Prognosis Assessment:**

Based on the available analysis, I can provide context from the treatment response prediction:

• **Predicted response**: {primary_pred.get('label', 'unknown').upper()}
• **Confidence**: {primary_pred.get('confidence', 0):.1%}

Patients with favorable treatment response typically have better outcomes. However, for detailed survival predictions, specific survival models would need to be run.

⚠️ This is a morphology-based assessment only. Comprehensive prognostic evaluation requires integration of clinical staging, molecular profiling, and multidisciplinary tumor board review."""
            
            return "Survival models not available for this analysis. Please run a full analysis first."
        
        # === TREATMENT RESPONSE QUESTIONS ===
        if any(word in q_lower for word in ["treatment", "response", "platinum", "chemo", "sensitive", "resistant"]):
            if "platinum_sensitivity" in context.predictions:
                pred = context.predictions["platinum_sensitivity"]
                if "error" in pred:
                    return f"Error in platinum sensitivity model: {pred.get('error')}"
                
                score = pred.get("score", 0.5)
                label = pred.get("label", "unknown")
                confidence = pred.get("confidence", 0)
                
                response = f"**Platinum Sensitivity Prediction:**\n\n"
                response += f"The model predicts **{label.upper()}** with {score:.1%} probability.\n\n"
                
                if confidence > 0.6:
                    response += "🟢 **High confidence** - The morphological patterns strongly suggest this classification.\n"
                elif confidence > 0.3:
                    response += "🟡 **Moderate confidence** - Consider pathologist review of high-attention regions.\n"
                else:
                    response += "🔴 **Low confidence** - Additional testing recommended before treatment decisions.\n"
                
                if context.similar_cases:
                    resp_count = sum(1 for c in context.similar_cases if c.get("label") == "responder")
                    non_resp = len(context.similar_cases) - resp_count
                    response += f"\n**Similar Cases:**\n"
                    response += f"• {resp_count} responders among {len(context.similar_cases)} similar cases\n"
                    response += f"• {non_resp} non-responders\n"
                
                return response
            
            return "Platinum sensitivity model not available. Please run analysis first."
        
        # === EVIDENCE / REGION QUESTIONS ===
        if any(word in q_lower for word in ["evidence", "region", "area", "attention", "patch", "show me", "where"]):
            if context.top_evidence:
                response = f"🔍 **High-Attention Regions ({len(context.top_evidence)} identified):**\n\n"
                for ev in context.top_evidence[:5]:
                    coords = ev.get("coordinates", [0, 0])
                    weight = ev["attention_weight"]
                    intensity = "High" if weight > 0.05 else "Moderate" if weight > 0.02 else "Low"
                    response += f"• **Region #{ev['rank']}** at ({coords[0]:,}, {coords[1]:,}) — {intensity} attention ({weight:.3f})\n"
                
                response += "\n📍 These regions contributed most to the model's prediction. "
                response += "Click on region buttons to navigate to them in the viewer.\n\n"
                response += "The attention weights indicate how much each tissue region influenced the classification."
                return response
            
            return "No high-attention regions identified. Run analysis first."
        
        # === SIMILAR CASES QUESTIONS ===
        if any(word in q_lower for word in ["similar", "compare", "other cases", "like this"]):
            if context.similar_cases:
                response = f"📊 **Similar Cases ({len(context.similar_cases)} found):**\n\n"
                
                responders = [c for c in context.similar_cases if c.get("label") == "responder"]
                non_responders = [c for c in context.similar_cases if c.get("label") == "non-responder"]
                
                response += f"• **{len(responders)}** were treatment responders\n"
                response += f"• **{len(non_responders)}** were non-responders\n\n"
                
                response += "**Top Matches:**\n"
                for c in context.similar_cases[:5]:
                    sim = c["similarity_score"]
                    label = c.get("label", "unknown")
                    emoji = "🟢" if label == "responder" else "🟠" if label == "non-responder" else "⚪"
                    response += f"• {emoji} `{c['slide_id']}` — {sim*100:.0f}% similarity ({label})\n"
                
                return response
            
            return "No similar cases found. The similarity search may not be available."
        
        # === WHY / REASONING QUESTIONS ===
        if any(word in q_lower for word in ["why", "reason", "explain", "how", "basis"]):
            response = "**Analysis Reasoning:**\n\n"
            
            if context.predictions:
                response += "**Model Predictions:**\n"
                for model_id, pred in context.predictions.items():
                    if "error" not in pred:
                        response += f"• {pred.get('model_name', model_id)}: {pred.get('label')} ({pred.get('score', 0):.1%})\n"
                response += "\n"
            
            if context.top_evidence:
                response += f"**Key Evidence:**\n"
                response += f"The model identified {len(context.top_evidence)} high-attention regions that drove the prediction. "
                response += "These tissue areas showed morphological patterns most associated with the predicted outcome.\n\n"
            
            if context.similar_cases:
                resp_count = sum(1 for c in context.similar_cases if c.get("label") == "responder")
                response += f"**Historical Context:**\n"
                response += f"Among {len(context.similar_cases)} morphologically similar cases, {resp_count} were responders. "
                response += "This provides additional support for the prediction.\n"
            
            return response
        
        # === CONFIDENCE QUESTIONS ===
        if any(word in q_lower for word in ["confidence", "certain", "reliable", "trust", "accurate"]):
            if context.predictions:
                response = "**Model Confidence Assessment:**\n\n"
                total_conf = 0
                count = 0
                
                for model_id, pred in context.predictions.items():
                    if "error" not in pred:
                        conf = pred.get("confidence", 0)
                        total_conf += conf
                        count += 1
                        
                        if conf > 0.6:
                            level = "🟢 HIGH"
                        elif conf > 0.3:
                            level = "🟡 MODERATE"
                        else:
                            level = "🔴 LOW"
                        response += f"• {pred.get('model_name', model_id)}: {level} ({conf:.1%})\n"
                
                if count > 0:
                    avg_conf = total_conf / count
                    response += f"\n**Overall Assessment:**\n"
                    if avg_conf > 0.6:
                        response += "High confidence overall. Predictions are suitable for clinical decision support."
                    elif avg_conf > 0.3:
                        response += "Moderate confidence. Consider pathologist review and additional testing."
                    else:
                        response += "Low confidence. Recommend molecular testing and expert consultation."
                
                return response
            
            return "Run analysis first to assess model confidence."
        
        # === REPORT QUESTIONS ===
        if any(word in q_lower for word in ["report", "summary", "findings", "conclusion"]):
            if context.report:
                return "The clinical report has been generated. Please check the Report panel for the full structured report with evidence and recommendations."
            
            response = "**Analysis Summary:**\n\n"
            
            if context.predictions:
                for model_id, pred in context.predictions.items():
                    if "error" not in pred:
                        response += f"• **{pred.get('model_name', model_id)}**: {pred.get('label')} ({pred.get('score', 0):.1%})\n"
            
            if context.top_evidence:
                response += f"\n• **Evidence**: {len(context.top_evidence)} high-attention regions identified\n"
            
            if context.similar_cases:
                resp = sum(1 for c in context.similar_cases if c.get("label") == "responder")
                response += f"• **Similar Cases**: {resp}/{len(context.similar_cases)} responders\n"
            
            response += "\nUse 'Generate Report' to create a full clinical report."
            return response
        
        # === DEFAULT RESPONSE ===
        return """I can help you understand this slide analysis. Try asking about:

• **Prognosis**: "What is the prognosis?"
• **Treatment response**: "What is the predicted platinum response?"
• **Evidence regions**: "Show me the high-attention areas"
• **Similar cases**: "How does this compare to similar cases?"
• **Reasoning**: "Why was this prediction made?"
• **Confidence**: "How confident is the model?"

Please select a slide and run analysis first if you haven't already."""
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        slide_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a chat message and yield response chunks.
        
        Implements RAG pattern:
        1. Get or create session
        2. Retrieve context if needed
        3. Generate response using context
        4. Update session history
        """
        # Get or create session
        session = self.get_or_create_session(session_id, slide_id)
        
        # Restore history if provided
        if history:
            for msg in history:
                if msg not in [{"role": m.role, "content": m.content} for m in session.history]:
                    session.add_message(msg["role"], msg["content"])
        
        # Add user message
        session.add_message("user", message)
        
        # Yield thinking step
        yield {
            "step": "thinking",
            "status": "running",
            "message": "Processing your question...",
        }
        
        # Retrieve context if slide specified
        if slide_id and (session.context is None or session.context.slide_id != slide_id):
            yield {
                "step": "retrieving",
                "status": "running",
                "message": f"Retrieving context for slide {slide_id}...",
            }
            session.context = await self.retrieve_context(slide_id)
        
        # Generate response
        if session.context:
            response = self._generate_response(message, session.context, session.history)
        else:
            response = "Please select a slide first to get contextual answers about the analysis."
        
        # Determine if we should include evidence patches
        evidence_patches = None
        q_lower = message.lower()
        if session.context and any(word in q_lower for word in ["region", "evidence", "show", "where", "area", "attention"]):
            evidence_patches = session.context.top_evidence[:5]
        
        # Add assistant message
        session.add_message("assistant", response, evidence_patches)
        
        # Yield final response
        yield {
            "step": "complete",
            "status": "complete",
            "reasoning": response,
            "evidence_patches": evidence_patches,
            "session_id": session.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
