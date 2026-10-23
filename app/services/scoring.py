from typing import Any

# Métrique principale utilisée pour le score de chaque type de problème
SCORE_METRIC = {
    "classification": "f1_weighted",
    "regression": "r2",
    "clustering": "silhouette",
}

SCORE_METRIC_LABEL = {
    "classification": "F1-score pondéré",
    "regression": "R² (ramené à 0 s'il est négatif)",
    "clustering": "Silhouette (ramenée à 0 si négative)",
}


def compute_score(problem_type: str, metrics: dict[str, Any]) -> float | None:
    """Renvoie un score de 0 à 100 (en %) pour un modèle.

    Le score est la métrique principale du problème, multipliée par 100.
    Les valeurs négatives sont ramenées à 0 : un modèle moins bon que le hasard
    obtient donc 0 %.
    """
    key = SCORE_METRIC.get(problem_type)
    if key is None:
        return None

    value = metrics.get(key)
    if value is None:
        return None

    value = float(value)
    if value != value:  # NaN
        return None

    percent = max(0.0, min(1.0, value)) * 100
    return round(percent, 1)