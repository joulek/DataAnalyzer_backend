from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

# JSONB sur PostgreSQL, JSON générique ailleurs (ex. SQLite)
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    rows: Mapped[int] = mapped_column(Integer)
    columns_count: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_pk: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    problem_type: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JsonType, default=dict)
    best_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    response: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset: Mapped["Dataset"] = relationship(back_populates="analyses")
    models: Mapped[list["ModelResult"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class ModelResult(Base):
    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_pk: Mapped[int] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(150))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # score de 0 à 100
    metrics: Mapped[dict] = mapped_column(JsonType, default=dict)
    charts: Mapped[list] = mapped_column(JsonType, default=list)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)

    analysis: Mapped["Analysis"] = relationship(back_populates="models")