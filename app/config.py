import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Informations de l'application ---
APP_NAME = "DataAnalyzer API"
APP_VERSION = "0.1.0"

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Règles d'upload ---
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

# --- CORS (adresses du frontend autorisées à appeler l'API) ---
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in _raw_origins.split(",")
    if origin.strip()
]
# --- Base de données ---
DATABASE_URL = os.getenv("DATABASE_URL", "")