from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Analysis

router = APIRouter(prefix="/api", tags=["Historique"])


@router.get("/history")
def list_history(db: Session = Depends(get_db)):
    analyses = db.scalars(
        select(Analysis).order_by(Analysis.created_at.desc()).limit(50)
    ).all()
    return [
        {
            "id": a.id,
            "dataset_id": a.dataset.dataset_id,
            "filename": a.dataset.filename,
            "target": a.target,
            "problem_type": a.problem_type,
            "best_model": a.best_model,
            "models_count": len(a.models),
            "created_at": a.created_at,
        }
        for a in analyses
    ]


@router.get("/history/{analysis_id}")
def get_history_item(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return analysis.response