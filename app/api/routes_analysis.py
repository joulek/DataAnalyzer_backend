import logging
import math
from typing import Any

from fastapi import APIRouter, Depends, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Analysis, Dataset, ModelResult
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, DetectionResult
from app.services.charts import (
    build_dataset_charts,
    build_supervised_charts,
    build_unsupervised_charts,
)
from app.services.dataset_loader import find_dataset_path, load_dataframe
from app.services.notebook import build_notebook_bytes
from app.services.preprocessing import prepare_supervised, prepare_unsupervised
from app.services.problem_detector import detect_problem_type
from app.services.report import build_report_pdf, summarize_dataset
from app.services.supervised import run_supervised
from app.services.unsupervised import run_unsupervised

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Analyse"])

# Ces routes sont déclarées avec "def" (et non "async def") : FastAPI les exécute
# dans un thread séparé, donc un entraînement long ne bloque pas le serveur.


class ReportRequest(BaseModel):
    """Requête d'export (PDF ou notebook) : la réponse complète de /analyze."""

    dataset_id: str
    filename: str = "dataset"
    analysis: dict[str, Any]


def _sanitize(obj: Any) -> Any:
    """Remplace NaN et inf par None.

    Un NaN n'est pas du JSON valide : Starlette refuse de le renvoyer au front
    et PostgreSQL refuse de le stocker dans une colonne JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _extract_models(results: Any) -> list[dict]:
    """Extrait la liste des modèles depuis `results`."""
    if isinstance(results, dict) and isinstance(results.get("models"), (list, dict)):
        results = results["models"]
    if isinstance(results, dict):
        return [
            {"key": k, **(v if isinstance(v, dict) else {"value": v})}
            for k, v in results.items()
        ]
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def _save_analysis(
    db: Session,
    request: AnalysisRequest,
    df,
    detection: dict,
    payload: dict,
) -> None:
    dataset = db.scalar(select(Dataset).where(Dataset.dataset_id == request.dataset_id))
    if dataset is None:  # dataset uploadé avant l'ajout de la base
        path = find_dataset_path(request.dataset_id)
        dataset = Dataset(
            dataset_id=request.dataset_id,
            filename=path.name,
            rows=len(df),
            columns_count=df.shape[1],
            file_path=str(path),
        )
        db.add(dataset)
        db.flush()

    results = payload.get("results") or {}
    best_key = results.get("best_model")

    analysis = Analysis(
        dataset_pk=dataset.id,
        target=detection.get("target"),
        problem_type=detection["problem_type"],
        config=_sanitize(jsonable_encoder(request)),
        best_model=best_key,
        response=payload,
    )

    for item in _extract_models(results):
        key = str(item.get("key") or item.get("name") or "modele")
        name = str(item.get("name") or item.get("model") or key)
        is_best = key == best_key

        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {
                k: v
                for k, v in item.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }

        # Score de 0 à 100, déjà nettoyé par _sanitize (float ou None)
        score = item.get("score")

        analysis.models.append(
            ModelResult(
                key=key[:100],
                name=name[:150],
                score=float(score) if score is not None else None,
                metrics=metrics,
                charts=[],
                is_best=is_best,
            )
        )

    db.add(analysis)
    db.commit()


@router.get("/datasets/{dataset_id}/detect", response_model=DetectionResult)
def detect_type(
    dataset_id: str,
    target: str | None = None,
    problem_type: str | None = None,
):
    """Propose le type de problème selon la colonne cible choisie (ou absente)."""
    df = load_dataframe(find_dataset_path(dataset_id))
    return detect_problem_type(df, target, problem_type)


@router.get("/datasets/{dataset_id}/charts")
def dataset_charts(dataset_id: str) -> dict[str, Any]:
    """Graphiques d'exploration : histogrammes, corrélations, valeurs manquantes."""
    df = load_dataframe(find_dataset_path(dataset_id))
    return _sanitize(build_dataset_charts(df))


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest, db: Session = Depends(get_db)):
    """Détecte le type de problème, entraîne les modèles et renvoie résultats + graphiques."""
    df = load_dataframe(find_dataset_path(request.dataset_id))

    detection = detect_problem_type(df, request.target, request.problem_type)

    if detection["learning_type"] == "supervised":
        X, y, dropped = prepare_supervised(df, detection["target"])
        results = run_supervised(X, y, detection["problem_type"], request.algorithms)
        charts = build_supervised_charts(results)
    else:
        X, dropped = prepare_unsupervised(df)
        results = run_unsupervised(X, request.n_clusters, request.algorithms)
        charts = build_unsupervised_charts(results)

    response = AnalysisResponse(
        dataset_id=request.dataset_id,
        detection=detection,
        dropped_columns=dropped,
        results=results,
        charts=charts,
    )

    # Une seule version nettoyée : identique pour le front et pour la base
    payload = _sanitize(jsonable_encoder(response))

    # Enregistrement en base (ne doit jamais faire échouer l'analyse)
    try:
        _save_analysis(db, request, df, detection, payload)
    except Exception:
        db.rollback()
        logger.exception("Enregistrement de l'analyse en base impossible")

    return payload


@router.post("/report")
def download_report(request: ReportRequest):
    """Génère le rapport PDF : données, prétraitement et résultats."""
    df = load_dataframe(find_dataset_path(request.dataset_id))
    target = (request.analysis.get("detection") or {}).get("target")

    summary = summarize_dataset(df, target)
    pdf = build_report_pdf(request.analysis, summary, request.filename)

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="rapport_{request.dataset_id[:8]}.pdf"'
        },
    )


@router.post("/notebook")
def download_notebook(request: ReportRequest):
    """Génère un notebook Jupyter qui reproduit l'analyse."""
    content = build_notebook_bytes(request.analysis, request.filename)

    return Response(
        content=content,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="analyse_{request.dataset_id[:8]}.ipynb"'
        },
    )