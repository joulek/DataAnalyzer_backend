from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload et aperçu du dataset
# ---------------------------------------------------------------------------
class ColumnInfo(BaseModel):
    """Informations sur une colonne du dataset."""

    name: str
    dtype: str
    is_numeric: bool
    missing: int
    unique: int


class DatasetPreview(BaseModel):
    """Aperçu du dataset renvoyé après l'upload."""

    rows: int
    columns_count: int
    columns: list[ColumnInfo]
    sample: list[dict[str, Any]]


class UploadResponse(BaseModel):
    """Réponse complète de la route d'upload."""

    dataset_id: str
    filename: str
    preview: DatasetPreview


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
class DetectionResult(BaseModel):
    """Résultat de la détection du type de problème."""

    learning_type: str  # "supervised" ou "unsupervised"
    problem_type: str  # "classification", "regression" ou "clustering"
    target: str | None = None
    reason: str
    n_classes: int | None = None
    warnings: list[str] = []


class AnalysisRequest(BaseModel):
    """Requête de lancement d'une analyse."""

    dataset_id: str
    target: str | None = None  # vide -> non supervisé
    problem_type: Literal["classification", "regression", "clustering"] | None = None
    algorithms: list[str] | None = None  # vide -> tous les algorithmes
    n_clusters: int | None = Field(default=None, ge=2)  # clustering uniquement


class DroppedColumn(BaseModel):
    column: str
    reason: str


class AnalysisResponse(BaseModel):
    """Réponse complète d'une analyse."""

    dataset_id: str
    detection: DetectionResult
    dropped_columns: list[DroppedColumn]
    results: dict[str, Any]  # sortie de run_supervised() ou run_unsupervised()
    charts: dict[str, Any]  # figures Plotly