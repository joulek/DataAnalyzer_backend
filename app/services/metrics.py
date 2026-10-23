import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def _round(value, digits: int = 4) -> float | None:
    """Arrondit une valeur. Renvoie None si elle est absente ou non finie (NaN, inf)."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _roc_auc(y_true, y_proba, labels: list[int]) -> float | None:
    """AUC ROC : binaire, ou one-vs-rest pour plus de deux classes.

    Renvoie None si elle ne peut pas être calculée (une classe absente du jeu de test,
    par exemple). Une métrique indisponible ne doit jamais faire échouer l'analyse.
    """
    try:
        if len(labels) == 2:
            return _round(roc_auc_score(y_true, y_proba[:, 1]))
        return _round(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", labels=labels, average="weighted"
            )
        )
    except ValueError:
        return None


def compute_classification_metrics(
    y_true,
    y_pred,
    class_names: list[str],
    y_proba=None,
) -> dict:
    """Métriques de classification : globales, par classe, accord et matrice de confusion.

    y_true et y_pred contiennent les classes encodées en entiers (0, 1, 2...).
    class_names[i] est le nom de la classe encodée i.
    y_proba (optionnel) : probabilités par classe, shape (n_échantillons, n_classes).
    Si elles sont fournies, l'AUC ROC est calculée.
    """
    labels = list(range(len(class_names)))

    # --- Moyennes globales ---
    # "weighted" : moyenne pondérée par la taille de chaque classe
    # "macro"    : moyenne simple, toutes les classes comptent pareil
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # --- Détail par classe ---
    p_c, r_c, f_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = [
        {
            "class": class_names[i],
            "precision": _round(p_c[i]),
            "recall": _round(r_c[i]),
            "f1": _round(f_c[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    ]

    # --- Matrice de confusion (lignes = réel, colonnes = prédit) ---
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _round(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": _round(p_w),
        "recall_weighted": _round(r_w),
        "f1_weighted": _round(f_w),
        "precision_macro": _round(p_m),
        "recall_macro": _round(r_m),
        "f1_macro": _round(f_m),
        # Accord au-delà du hasard : 0 = hasard, 1 = accord parfait
        "cohen_kappa": _round(cohen_kappa_score(y_true, y_pred, labels=labels)),
        # Corrélation entre réel et prédit, robuste aux classes déséquilibrées
        "mcc": _round(matthews_corrcoef(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": matrix.tolist(),
        },
    }

    if y_proba is not None:
        metrics["roc_auc"] = _roc_auc(y_true, np.asarray(y_proba), labels)

    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Métriques de régression : erreurs, R², variance expliquée et erreur relative."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    # MAPE : erreur relative en %. Calculée uniquement sur les valeurs réelles non nulles,
    # car une division par zéro rendrait la métrique inutilisable.
    non_zero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        if non_zero.any()
        else None
    )

    return {
        "mae": _round(mean_absolute_error(y_true, y_pred)),
        "mse": _round(mse),
        "rmse": _round(np.sqrt(mse)),
        "r2": _round(r2_score(y_true, y_pred)),
        "explained_variance": _round(explained_variance_score(y_true, y_pred)),
        # Erreur médiane : moins sensible aux valeurs extrêmes que la MAE
        "median_ae": _round(median_absolute_error(y_true, y_pred)),
        "mape": _round(mape, 2),
    }