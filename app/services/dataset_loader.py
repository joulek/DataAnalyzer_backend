import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, UPLOAD_DIR


def validate_extension(filename: str) -> str:
    """Vérifie l'extension du fichier et la retourne en minuscules."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés : {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return extension


async def save_upload(file: UploadFile) -> tuple[str, Path]:
    """Sauvegarde le fichier uploadé sous un identifiant unique.

    Retourne (dataset_id, chemin du fichier).
    """
    extension = validate_extension(file.filename or "")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (maximum {MAX_FILE_SIZE_BYTES // (1024 * 1024)} Mo).",
        )

    dataset_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{dataset_id}{extension}"
    file_path.write_bytes(content)

    return dataset_id, file_path


def find_dataset_path(dataset_id: str) -> Path:
    """Retrouve le fichier d'un dataset à partir de son identifiant."""
    # Sécurité : l'identifiant doit être strictement hexadécimal
    if not dataset_id.isalnum():
        raise HTTPException(status_code=400, detail="Identifiant de dataset invalide.")

    for extension in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{dataset_id}{extension}"
        if candidate.exists():
            return candidate

    raise HTTPException(status_code=404, detail="Dataset introuvable.")


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Charge le fichier en DataFrame pandas."""
    try:
        if file_path.suffix.lower() == ".csv":
            # sep=None + engine="python" : détecte automatiquement , ou ;
            df = pd.read_csv(file_path, sep=None, engine="python")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Impossible de lire le fichier. Vérifiez qu'il est valide.",
        )

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Le dataset doit contenir au moins 2 colonnes et 1 ligne.",
        )

    return df


def get_dataset_preview(df: pd.DataFrame, rows: int = 5) -> dict:
    """Construit l'aperçu renvoyé au frontend après l'upload."""
    columns = []
    for col in df.columns:
        series = df[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )

    # to_json puis json.loads garantit des valeurs sérialisables (NaN -> null)
    import json

    sample = json.loads(df.head(rows).to_json(orient="records"))

    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": columns,
        "sample": sample,
    }