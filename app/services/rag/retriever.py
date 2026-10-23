"""Recherche des passages les plus proches d'une question."""

from app.services.rag.config import TOP_K
from app.services.rag.index import get_collection, get_embedder


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Renvoie les passages les plus pertinents, avec leur score de similarité."""
    question = question.strip()
    if not question:
        return []

    vector = get_embedder().encode([question], normalize_embeddings=True)[0]
    result = get_collection().query(
        query_embeddings=[vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        passages.append(
            {
                "text": doc,
                "source": meta["source"],
                "section": meta["section"],
                # Distance cosinus -> similarité (1 = identique)
                "score": round(1 - float(dist), 3),
            }
        )
    return passages