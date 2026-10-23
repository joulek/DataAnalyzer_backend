"""Pose quelques questions au chatbot complet. À lancer depuis backend/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer import answer_question  # noqa: E402

QUESTIONS = [
    "Comment est calculé le score ?",
    "How is the score calculated?",
    "Quelle est la limite du SVM dans DataAnalyzer ?",
    "Quel est le meilleur film de tous les temps ?",  # hors sujet : doit refuser
]

if __name__ == "__main__":
    for question in QUESTIONS:
        result = answer_question(question)
        print(f"\n? {question}\n")
        print(result["answer"])
        print("\nSources :", ", ".join(s["source"] for s in result["sources"]))