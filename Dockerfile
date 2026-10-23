FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Dépendance système requise par scikit-learn et torch
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libpq5 \
 && rm -rf /var/lib/apt/lists/*

# 1) PyTorch version CPU (beaucoup plus léger que la version CUDA par défaut)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2) Dépendances du projet
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Modèle d'embeddings téléchargé à la construction, pas au premier appel
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 4) Code et données
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY chroma_db ./chroma_db

# Utilisateur non root, dossiers inscriptibles (uploads, feedback, index)
RUN useradd --create-home appuser \
 && mkdir -p /app/uploads \
 && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]