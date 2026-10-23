"""Reconstruit l'index à partir de data/knowledge/. À lancer depuis backend/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.index import build_index  # noqa: E402

if __name__ == "__main__":
    count = build_index()
    print(f"Index reconstruit : {count} passages indexés.")