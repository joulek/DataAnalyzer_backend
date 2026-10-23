import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.schemas.analysis import DatasetPreview, UploadResponse
from app.services.dataset_loader import (
    find_dataset_path,
    get_dataset_preview,
    load_dataframe,
    save_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Dataset"])


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Reçoit un dataset (CSV ou XLSX), le sauvegarde et renvoie un aperçu."""
    dataset_id, file_path = await save_upload(file)

    try:
        df = load_dataframe(file_path)
    except HTTPException:
        # Fichier illisible ou invalide : on ne le garde pas sur le disque
        file_path.unlink(missing_ok=True)
        raise

    filename = file.filename or file_path.name

    # Enregistrement en base (ne doit jamais faire échouer l'upload)
    try:
        db.add(
            Dataset(
                dataset_id=dataset_id,
                filename=filename,
                rows=len(df),
                columns_count=df.shape[1],
                file_path=str(file_path),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Enregistrement du dataset en base impossible")

    return UploadResponse(
        dataset_id=dataset_id,
        filename=filename,
        preview=get_dataset_preview(df),
    )


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def get_preview(dataset_id: str):
    """Renvoie l'aperçu d'un dataset déjà uploadé."""
    file_path = find_dataset_path(dataset_id)
    df = load_dataframe(file_path)
    return get_dataset_preview(df)