"""Configuration du RAG : chemins, modèles et paramètres de recherche et de génération."""

from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[3]  # dossier backend/
load_dotenv(BASE_DIR / ".env")

# --- Base de connaissances et index ---
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "dataanalyzer_knowledge"

# --- Recherche ---
# Modèle d'embeddings multilingue (français et anglais), exécuté en local
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_CHUNK_CHARS = 1200  # taille maximale d'un passage, en caractères
TOP_K = 3               # nombre de passages renvoyés par la recherche

# --- Génération (LLM) ---
LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-120b"  # vérifiez le nom exact dans la console Groq
LLM_MAX_TOKENS = 400
ANSWER_TEMPERATURE = 0.2  # faible : réponses stables, peu d'invention
RAG_MIN_SCORE = 0.50