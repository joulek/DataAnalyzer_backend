"""Construction et ouverture de l'index vectoriel (ChromaDB)."""

import chromadb
from sentence_transformers import SentenceTransformer

from app.services.rag.chunking import chunk_knowledge_base
from app.services.rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, KNOWLEDGE_DIR

_model: SentenceTransformer | None = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    """Charge le modèle d'embeddings une seule fois pour tout le processus."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    """Ouvre la collection une seule fois : pas de reconnexion à chaque question."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def warm_up() -> None:
    """Charge le modèle et la collection au démarrage, pour que la première question soit rapide."""
    get_embedder()
    get_collection()


def build_index() -> int:
    """Reconstruit l'index à partir du dossier de connaissances. Renvoie le nombre de passages."""
    global _collection
    chunks = chunk_knowledge_base(KNOWLEDGE_DIR)
    if not chunks:
        raise RuntimeError(f"Aucun document .md trouvé dans {KNOWLEDGE_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    vectors = get_embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[
            {"source": c["source"], "title": c["title"], "section": c["section"]}
            for c in chunks
        ],
    )
    _collection = None  # force la réouverture au prochain appel
    return len(chunks)