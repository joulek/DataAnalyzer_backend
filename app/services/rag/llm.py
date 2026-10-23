"""Appel au fournisseur de LLM (Groq). Seul ce fichier dépend du fournisseur."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from app.services.rag.config import ANSWER_TEMPERATURE, LLM_MAX_TOKENS, LLM_MODEL
from app.services.rag.rate_limiter import wait_before_llm_call

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MAX_RETRIES = 1
BASE_DELAY_SECONDS = 2


def generate(system: str, user: str, history: list[dict] | None = None) -> str:
    """Envoie les consignes, les échanges précédents et la question, renvoie la réponse."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY n'est pas définie dans le fichier .env du backend.")

    messages = [{"role": "system", "content": system}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history or []]
    messages.append({"role": "user", "content": user})

    client = Groq(api_key=api_key)

    for attempt in range(MAX_RETRIES + 1):
        wait_before_llm_call()
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=ANSWER_TEMPERATURE,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as exc:
            message = str(exc)
            rate_limited = "429" in message or "rate" in message.lower()
            if not rate_limited or attempt == MAX_RETRIES:
                raise RuntimeError(f"Erreur du service de génération : {message}") from exc
            time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))

    raise RuntimeError("Réponse indisponible après plusieurs tentatives.")

def generate_stream(system: str, user: str, history: list[dict] | None = None):
    """Comme generate, mais renvoie les morceaux de texte au fur et à mesure (générateur)."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY n'est pas définie dans le fichier .env du backend.")

    messages = [{"role": "system", "content": system}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history or []]
    messages.append({"role": "user", "content": user})

    wait_before_llm_call()
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece