import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.rag.analysis_context import summarize_analysis
from app.services.rag.answer import answer_question
from app.services.rag.rate_limiter import RateLimitExceeded, check_client_quota
import json
from fastapi.responses import StreamingResponse
from app.services.rag.answer import stream_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Assistant"])

MAX_QUESTION_CHARS = 500
MAX_HISTORY_TURNS = 6
# Les réponses de l'assistant peuvent être longues : on accepte plus à l'entrée,
# puis on tronque avant d'envoyer l'historique au LLM
MAX_INCOMING_TURN_CHARS = 5000
MAX_TURN_CHARS = 1000


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_INCOMING_TURN_CHARS)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=MAX_QUESTION_CHARS)
    analysis: dict[str, Any] | None = None  # réponse de /analyze, optionnelle
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


@router.post("/chat")
def chat(request_body: ChatRequest, request: Request):
    """Répond à une question, en tenant compte des échanges précédents."""
    client_id = request.client.host if request.client else "inconnu"
    try:
        check_client_quota(client_id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    summary = summarize_analysis(request_body.analysis) if request_body.analysis else None

    # Historique borné avant d'atteindre le LLM
    history = [
        {"role": turn.role, "content": turn.content[:MAX_TURN_CHARS]}
        for turn in request_body.history
    ]

    try:
        return answer_question(request_body.question.strip(), summary, history)
    except RuntimeError as exc:
        logger.exception("Erreur de l'assistant : %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc




@router.post("/chat/stream")
def chat_stream(request_body: ChatRequest, request: Request):
    """Réponse en flux (Server-Sent Events), même logique que /chat."""
    client_id = request.client.host if request.client else "inconnu"
    try:
        check_client_quota(client_id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    summary = summarize_analysis(request_body.analysis) if request_body.analysis else None
    history = [
        {"role": turn.role, "content": turn.content[:MAX_TURN_CHARS]}
        for turn in request_body.history
    ]

    def events():
        try:
            for event in stream_answer(request_body.question.strip(), summary, history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            logger.exception("Erreur de l'assistant (flux) : %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")