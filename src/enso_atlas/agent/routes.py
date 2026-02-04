"""
Enso Atlas Agent API Routes - Streaming endpoints for agentic workflows.

Provides:
- POST /api/agent/analyze - Start multi-step agent analysis with SSE streaming
- POST /api/agent/followup - Ask follow-up questions about an analysis
- GET /api/agent/session/{session_id} - Get session state and results
- GET /api/agent/sessions - List active sessions
"""

from pathlib import Path
from typing import Optional, List
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent Workflow"])


# Request/Response Models
class AgentAnalyzeRequest(BaseModel):
    """Request to start agent analysis."""
    slide_id: str = Field(..., min_length=1, max_length=256)
    clinical_context: str = Field(
        default="",
        max_length=2000,
        description="Clinical context (e.g., '58yo female, FIGO stage IIIC, post-debulking')"
    )
    questions: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific questions to answer (e.g., 'What is the predicted treatment response?')"
    )


class AgentFollowupRequest(BaseModel):
    """Request for follow-up question on existing analysis."""
    session_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=1000)


class AgentSessionResponse(BaseModel):
    """Response with session information."""
    session_id: str
    slide_id: str
    current_step: str
    created_at: str
    has_report: bool
    error: Optional[str] = None


class AgentSessionListResponse(BaseModel):
    """Response listing active sessions."""
    sessions: List[AgentSessionResponse]
    total: int


# Global agent workflow instance (set by main.py)
_agent_workflow = None


def set_agent_workflow(workflow):
    """Set the global agent workflow instance."""
    global _agent_workflow
    _agent_workflow = workflow


def get_agent_workflow():
    """Get the global agent workflow instance."""
    if _agent_workflow is None:
        raise HTTPException(
            status_code=503,
            detail="Agent workflow not initialized"
        )
    return _agent_workflow


async def stream_agent_analysis(slide_id: str, clinical_context: str, questions: List[str]):
    """
    Generator that streams agent workflow results as Server-Sent Events (SSE).
    
    Each event is formatted as:
    data: {"step": "...", "status": "...", "message": "...", ...}
    
    """
    workflow = get_agent_workflow()
    
    try:
        async for step_result in workflow.run_workflow(
            slide_id=slide_id,
            clinical_context=clinical_context,
            questions=questions,
        ):
            # Format as SSE event
            event_data = {
                "step": step_result.step.value,
                "status": step_result.status.value,
                "message": step_result.message,
                "reasoning": step_result.reasoning,
                "data": step_result.data,
                "timestamp": step_result.timestamp,
                "duration_ms": step_result.duration_ms,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            
    except Exception as e:
        logger.exception(f"Agent analysis error: {e}")
        error_event = {
            "step": "error",
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "data": {"error": str(e)},
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/analyze")
async def start_agent_analysis(request: AgentAnalyzeRequest):
    """
    Start multi-step agent analysis with streaming results.
    
    Returns Server-Sent Events (SSE) stream with progress updates.
    Each event contains:
    - step: Current workflow step (initialize, analyze, retrieve, compare, reason, report)
    - status: Step status (running, complete, error)
    - message: Human-readable progress message
    - reasoning: Agent's reasoning for this step
    - data: Step-specific data (predictions, similar cases, etc.)
    
    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/api/agent/analyze?...');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.step, data.message);
    };
    ```
    
    Or with fetch:
    ```javascript
    const response = await fetch('/api/agent/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slide_id: '...', clinical_context: '...' })
    });
    const reader = response.body.getReader();
    // Process SSE stream...
    ```
    """
    workflow = get_agent_workflow()
    
    return StreamingResponse(
        stream_agent_analysis(
            request.slide_id,
            request.clinical_context,
            request.questions,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


async def stream_followup(session_id: str, question: str):
    """Generator for follow-up question stream."""
    workflow = get_agent_workflow()
    
    try:
        async for step_result in workflow.followup(session_id, question):
            event_data = {
                "step": step_result.step.value,
                "status": step_result.status.value,
                "message": step_result.message,
                "reasoning": step_result.reasoning,
                "data": step_result.data,
                "timestamp": step_result.timestamp,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            
    except Exception as e:
        logger.exception(f"Follow-up error: {e}")
        error_event = {
            "step": "error",
            "status": "error",
            "message": f"Follow-up failed: {str(e)}",
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/followup")
async def agent_followup(request: AgentFollowupRequest):
    """
    Ask a follow-up question about an existing analysis.
    
    The agent remembers context from the previous analysis and can answer
    questions like:
    - "Why did you predict platinum resistant?"
    - "Show me the regions driving that prediction"
    - "What is the survival prognosis?"
    - "How confident is the model?"
    
    Returns SSE stream with the agent's response.
    """
    workflow = get_agent_workflow()
    
    # Verify session exists
    state = workflow.get_session(request.session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    return StreamingResponse(
        stream_followup(request.session_id, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get detailed information about an agent session.
    
    Returns the full session state including:
    - Analysis results and predictions
    - Similar cases found
    - Evidence patches
    - Generated report
    - Conversation history
    - Full reasoning chain
    """
    workflow = get_agent_workflow()
    
    state = workflow.get_session(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    result = workflow.get_result(session_id)
    
    return {
        "session_id": state.session_id,
        "created_at": state.created_at,
        "slide_id": state.context.slide_id if state.context else None,
        "clinical_context": state.context.clinical_context if state.context else None,
        "questions": state.context.questions if state.context else [],
        "current_step": state.current_step.value,
        "error": state.error,
        
        # Analysis results
        "predictions": result.predictions if result else {},
        "similar_cases": result.similar_cases if result else [],
        "top_evidence": result.top_evidence if result else [],
        "report": result.report if result else None,
        "reasoning_chain": result.reasoning_chain if result else [],
        
        # Conversation
        "conversation_history": state.conversation_history,
        
        # Timing
        "total_duration_ms": result.total_duration_ms if result else 0,
    }


@router.get("/sessions", response_model=AgentSessionListResponse)
async def list_sessions(limit: int = 20, offset: int = 0):
    """
    List active agent sessions.
    
    Returns summary information about recent analysis sessions.
    """
    workflow = get_agent_workflow()
    
    sessions = []
    for session_id, state in list(workflow._sessions.items())[offset:offset+limit]:
        sessions.append(AgentSessionResponse(
            session_id=state.session_id,
            slide_id=state.context.slide_id if state.context else "unknown",
            current_step=state.current_step.value,
            created_at=state.created_at,
            has_report=state.report is not None,
            error=state.error,
        ))
    
    return AgentSessionListResponse(
        sessions=sessions,
        total=len(workflow._sessions),
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete an agent session.
    
    Removes the session and frees associated memory.
    """
    workflow = get_agent_workflow()
    
    if session_id not in workflow._sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    del workflow._sessions[session_id]
    
    return {"success": True, "message": f"Session {session_id} deleted"}
