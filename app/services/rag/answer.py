"""Point d'entrée du chatbot : RAG si la base répond, sinon LLM général, avec historique."""

from app.services.rag.config import RAG_MIN_SCORE
from app.services.rag.llm import generate
from app.services.rag.prompts import (
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_RAG,
    build_general_message,
    build_rag_message,
)
from app.services.rag.retriever import retrieve
from app.services.rag.llm import generate_stream


# Une question plus courte que ce seuil est considérée comme une question de suivi
FOLLOW_UP_MAX_CHARS = 60


def _retrieval_query(question: str, history: list[dict]) -> str:
    """Pour une question de suivi, ajoute la dernière question de l'utilisateur à la recherche."""
    if history and len(question) < FOLLOW_UP_MAX_CHARS:
        last_user = next(
            (turn["content"] for turn in reversed(history) if turn["role"] == "user"),
            None,
        )
        if last_user:
            return f"{last_user} {question}"
    return question


def answer_question(
    question: str,
    analysis_summary: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Renvoie la réponse, le mode utilisé (rag ou general) et les sources éventuelles."""
    history = history or []
    passages = retrieve(_retrieval_query(question, history))
    best_score = passages[0]["score"] if passages else 0.0

    if passages and best_score >= RAG_MIN_SCORE:
        text = generate(
            SYSTEM_PROMPT_RAG,
            build_rag_message(question, passages, analysis_summary),
            history,
        )
        return {
            "answer": text,
            "mode": "rag",
            "sources": [
                {"index": i, "source": p["source"], "section": p["section"], "score": p["score"]}
                for i, p in enumerate(passages, start=1)
            ],
        }

    text = generate(
        SYSTEM_PROMPT_GENERAL,
        build_general_message(question, analysis_summary),
        history,
    )
    return {"answer": text, "mode": "general", "sources": []}




def stream_answer(question: str, analysis_summary: str | None = None, history: list[dict] | None = None):
    """Renvoie des événements : d'abord le mode et les sources, puis les morceaux de texte."""
    history = history or []
    passages = retrieve(_retrieval_query(question, history))
    best_score = passages[0]["score"] if passages else 0.0

    if passages and best_score >= RAG_MIN_SCORE:
        sources = [
            {"index": i, "source": p["source"], "section": p["section"], "score": p["score"]}
            for i, p in enumerate(passages, start=1)
        ]
        yield {"type": "meta", "mode": "rag", "sources": sources}
        stream = generate_stream(
            SYSTEM_PROMPT_RAG,
            build_rag_message(question, passages, analysis_summary),
            history,
        )
    else:
        yield {"type": "meta", "mode": "general", "sources": []}
        stream = generate_stream(
            SYSTEM_PROMPT_GENERAL,
            build_general_message(question, analysis_summary),
            history,
        )

    for piece in stream:
        yield {"type": "token", "text": piece}
    yield {"type": "done"}