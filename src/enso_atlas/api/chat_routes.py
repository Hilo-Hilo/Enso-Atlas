"""RAG chat API routes."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000)
    slide_id: str | None = Field(None, max_length=256)
    session_id: str | None = Field(None, max_length=64)
    history: list[dict[str, str]] | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    response: str
    session_id: str
    evidence_patches: list[dict[str, Any]] | None = None


def create_chat_router(chat_manager_provider: Callable[[], Any]) -> APIRouter:
    """Create chat routes that resolve ChatManager lazily after startup."""
    router = APIRouter()

    def _chat_manager():
        manager = chat_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="Chat manager not initialized")
        return manager

    async def stream_chat(
        message: str, slide_id: str | None, session_id: str | None, history: list | None
    ):
        """Stream chat responses as SSE."""
        async for result in _chat_manager().chat(
            message=message,
            session_id=session_id,
            slide_id=slide_id,
            history=history,
        ):
            yield f"data: {json.dumps(result)}\n\n"

    @router.post("/api/chat")
    async def chat_endpoint(request: ChatRequest):
        """RAG-based chat endpoint for conversational AI assistant."""
        return StreamingResponse(
            stream_chat(
                request.message,
                request.slide_id,
                request.session_id,
                request.history,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/chat/session/{session_id}")
    async def get_chat_session(session_id: str):
        """Get chat session history and context."""
        session = _chat_manager()._sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {
            "session_id": session.session_id,
            "slide_id": session.slide_id,
            "created_at": session.created_at,
            "history": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "evidence_patches": msg.evidence_patches,
                }
                for msg in session.history
            ],
            "has_context": session.context is not None,
        }

    return router
