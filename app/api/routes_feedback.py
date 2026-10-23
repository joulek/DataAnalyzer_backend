import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["Assistant"])

FEEDBACK_FILE = Path(__file__).resolve().parents[2] / "data" / "feedback.jsonl"


class FeedbackRequest(BaseModel):
    question: str = Field(max_length=500)
    answer: str = Field(max_length=5000)
    mode: Literal["rag", "general"]
    vote: Literal["up", "down"]


@router.post("/feedback")
def save_feedback(body: FeedbackRequest):
    """Enregistre un vote dans data/feedback.jsonl (une ligne JSON par vote)."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(), **body.model_dump()}
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}