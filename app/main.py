from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401  (enregistre les tables auprès de SQLAlchemy)
from app.api.routes_analysis import router as analysis_router
from app.api.routes_chat import router as chat_router
from app.api.routes_upload import router as upload_router
from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from app.database import Base, engine
from app.services.rag.index import warm_up
from app.api.routes_feedback import router as feedback_router



app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API d'analyse automatique de datasets avec Machine Learning.",
)

# --- Base de données : création des tables si elles n'existent pas ---
Base.metadata.create_all(bind=engine)
@app.on_event("startup")
def preload_rag():
    try:
        warm_up()
    except Exception:
        # Le RAG reste optionnel : l'analyse fonctionne même si l'index manque
        pass

# --- CORS : autorise le frontend à appeler l'API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(chat_router)
app.include_router(feedback_router)


@app.get("/", tags=["Santé"])
def root():
    return {"message": f"{APP_NAME} est en ligne", "version": APP_VERSION}


@app.get("/health", tags=["Santé"])
def health_check():
    return {"status": "ok"}