import time

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from app.services.preprocessing import build_preprocessor
from app.services.scoring import compute_score

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_PREDICTIONS_RETURNED = 500
# Les SVM ont un coût quadratique : on ne les entraîne que sur un nombre raisonnable de lignes
MAX_SVM_TRAIN_ROWS = 5000

# Fabriques de modèles : on crée une instance neuve à chaque entraînement
CLASSIFICATION_MODELS = {
    "logistic_regression": (
        "Régression logistique",
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "random_forest": (
        "Random Forest",
        lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "decision_tree": (
        "Arbre de décision",
        lambda: DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "knn": (
        "K plus proches voisins",
        lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
    ),
    "svm": (
        "SVM (noyau RBF)",
        lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
    ),
}

REGRESSION_MODELS = {
    "linear_regression": ("Régression linéaire", lambda: LinearRegression()),
    "random_forest": (
        "Random Forest",
        lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    ),
    "decision_tree": (
        "Arbre de décision",
        lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    ),
    "knn": (
        "K plus proches voisins",
        lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
    ),
    "svm": (
        "SVM (noyau RBF)",
        lambda: SVR(kernel="rbf", C=1.0),
    ),
}

# Algorithmes soumis à une limite de lignes d'entraînement
ROW_LIMITS = {"svm": MAX_SVM_TRAIN_ROWS}

# Métrique utilisée pour désigner le meilleur modèle
BEST_METRIC = {"classification": "f1_weighted", "regression": "r2"}


def _top_feature_importance(pipeline: Pipeline, limit: int = 10) -> list[dict]:
    """Importance des variables pour les modèles qui l'exposent (arbres, forêts, boosting)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    names = [str(name).split("__", 1)[-1] for name in names]
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": names[i], "importance": round(float(importances[i]), 4)}
        for i in order
    ]


def _prepare_classification_target(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Retire les classes à une seule observation et encode la cible en entiers."""
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    mask = y.isin(valid_classes)
    X, y = X[mask], y[mask]

    if y.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Il reste moins de 2 classes exploitables après nettoyage.",
        )

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = [str(c) for c in encoder.classes_]

    return X, y_encoded, class_names


def run_supervised(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    algorithms: list[str] | None = None,
) -> dict:
    """Entraîne plusieurs modèles et renvoie leurs métriques et leur score (0 à 100).

    problem_type : "classification" ou "regression".
    algorithms   : liste de clés de modèles (par défaut, tous).
    """
    if problem_type not in BEST_METRIC:
        raise HTTPException(status_code=400, detail="Type de problème supervisé invalide.")

    catalog = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS

    selected = algorithms or list(catalog.keys())
    unknown = [key for key in selected if key not in catalog]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithme(s) inconnu(s) : {', '.join(unknown)}. "
            f"Disponibles : {', '.join(catalog.keys())}",
        )

    # --- Préparation de la cible ---
    class_names = None
    if problem_type == "classification":
        X, y_values, class_names = _prepare_classification_target(X, y)
    else:
        y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_values).any():
            raise HTTPException(
                status_code=400,
                detail="La cible de régression contient des valeurs non numériques.",
            )

    # --- Découpage train / test ---
    stratify = None
    if problem_type == "classification":
        n_test = int(np.ceil(len(y_values) * TEST_SIZE))
        if n_test >= len(class_names):
            stratify = y_values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # --- Entraînement de chaque modèle ---
    results = []
    errors = []

    for key in selected:
        label, factory = catalog[key]

        # Garde-fou : certains algorithmes deviennent trop lents sur de grands jeux
        limit = ROW_LIMITS.get(key)
        if limit is not None and len(X_train) > limit:
            errors.append(
                {
                    "model": label,
                    "error": f"dataset trop grand pour cet algorithme (max {limit} lignes d'entraînement)",
                }
            )
            continue

        # Le préprocesseur est dans le pipeline : il est entraîné uniquement sur X_train
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", factory()),
            ]
        )

        try:
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            y_pred = pipeline.predict(X_test)
        except Exception as exc:
            errors.append({"model": label, "error": str(exc)})
            continue

        entry = {
            "key": key,
            "name": label,
            "train_time_s": round(train_time, 3),
            "feature_importance": _top_feature_importance(pipeline),
        }

        if problem_type == "classification":
            entry["metrics"] = compute_classification_metrics(y_test, y_pred, class_names)
        else:
            entry["metrics"] = compute_regression_metrics(y_test, y_pred)
            entry["predictions"] = {
                "y_true": [round(float(v), 4) for v in y_test[:MAX_PREDICTIONS_RETURNED]],
                "y_pred": [round(float(v), 4) for v in y_pred[:MAX_PREDICTIONS_RETURNED]],
            }

        # Score de 0 à 100 %, calculé à partir de la métrique principale
        entry["score"] = compute_score(problem_type, entry["metrics"])

        results.append(entry)

    if not results:
        raise HTTPException(
            status_code=500,
            detail=f"Aucun modèle n'a pu être entraîné. Détails : {errors}",
        )

    # --- Meilleur modèle ---
    metric_name = BEST_METRIC[problem_type]
    best = max(results, key=lambda r: r["metrics"][metric_name])

    return {
        "problem_type": problem_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "class_names": class_names,
        "models": results,
        "failed_models": errors,
        "best_model": best["key"],
        "best_metric": metric_name,
        "best_score": best["score"],
    }