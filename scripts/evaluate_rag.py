"""Évalue le RAG sur data/eval/questions.json : rappel, routage et fidélité des citations.
À lancer depuis backend/."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer import answer_question  # noqa: E402
from app.services.rag.retriever import retrieve  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.json"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, passages_count: int) -> bool:
    """Vrai si toutes les citations [n] désignent un passage fourni."""
    numbers = [int(n) for n in CITATION_PATTERN.findall(answer)]
    return all(1 <= n <= passages_count for n in numbers)


def main() -> None:
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    retrieval_top1 = retrieval_topk = routing_ok = citations_ok = citation_cases = 0
    rag_cases = 0

    for case in cases:
        question = case["question"]
        expected_source = case["expected_source"]
        expected_mode = case["expected_mode"]

        # 1. Recherche seule : le bon document est-il retrouvé ?
        passages = retrieve(question, top_k=3)
        best_score = passages[0]["score"] if passages else 0.0
        sources = [p["source"] for p in passages]
        if expected_source:
            rag_cases += 1
            retrieval_top1 += bool(sources) and sources[0] == expected_source
            retrieval_topk += expected_source in sources

        # 2. Chaîne complète : routage et citations
        result = answer_question(question)
        routing_ok += result["mode"] == expected_mode

        if result["mode"] == "rag":
            citation_cases += 1
            citations_ok += check_citations(result["answer"], len(result["sources"]))

        status = "OK" if result["mode"] == expected_mode else "KO"
        print(f"[{status}] {question}")
        print(f"      mode : {result['mode']} (attendu : {expected_mode})")
        print(f"      sources retrouvées : {sources}")
        print(f"      meilleur score : {best_score}")

    total = len(cases)
    print("\n=== Résultats ===")
    print(f"Rappel top-1 (bon document en tête) : {retrieval_top1}/{rag_cases}")
    print(f"Rappel top-3 (bon document dans les 3) : {retrieval_topk}/{rag_cases}")
    print(f"Routage correct : {routing_ok}/{total}")
    print(f"Citations valides : {citations_ok}/{citation_cases}")


if __name__ == "__main__":
    main()