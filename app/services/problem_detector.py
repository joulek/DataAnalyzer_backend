import pandas as pd
from fastapi import HTTPException

VALID_TYPES = {"classification", "regression", "clustering"}

# Une cible numérique avec peu de valeurs différentes est traitée comme des classes
CLASSIFICATION_MAX_UNIQUE = 10
# Cible entière : jusqu'à 20 valeurs, si elles sont rares par rapport au nombre de lignes
INTEGER_MAX_UNIQUE = 20
INTEGER_RATIO_THRESHOLD = 0.05
# Au-delà, une cible catégorielle n'a plus de sens pour une classification
MAX_CLASSES = 50
# Rapport minimal entre la plus petite et la plus grande classe avant d'avertir
IMBALANCE_THRESHOLD = 0.1


def _is_integer_like(series: pd.Series) -> bool:
    """Vrai si la série ne contient que des nombres entiers (même stockés en float)."""
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        return bool((series == series.round()).all())
    return False


def detect_problem_type(
    df: pd.DataFrame,
    target: str | None = None,
    forced_type: str | None = None,
) -> dict:
    """Détermine le type de problème ML.

    - Pas de colonne cible  -> non supervisé (clustering).
    - Cible catégorielle    -> classification.
    - Cible numérique       -> régression (ou classification si peu de valeurs).

    `forced_type` permet à l'utilisateur de corriger la détection automatique.
    """
    if forced_type is not None and forced_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Type de problème invalide. Valeurs acceptées : {', '.join(sorted(VALID_TYPES))}",
        )

    target = target.strip() if target else None

    # --- Cas 1 : aucune cible -> non supervisé ---
    if not target:
        if forced_type in {"classification", "regression"}:
            raise HTTPException(
                status_code=400,
                detail="Une classification ou une régression nécessite une colonne cible.",
            )
        return {
            "learning_type": "unsupervised",
            "problem_type": "clustering",
            "target": None,
            "reason": "Aucune colonne cible choisie : analyse non supervisée (clustering).",
            "n_classes": None,
            "warnings": [],
        }

    # --- Cas 2 : une cible est fournie -> supervisé ---
    if forced_type == "clustering":
        raise HTTPException(
            status_code=400,
            detail="Le clustering n'utilise pas de colonne cible. Retirez la cible ou changez de type.",
        )

    columns = [str(col).strip() for col in df.columns]
    if target not in columns:
        raise HTTPException(status_code=400, detail=f"La colonne cible '{target}' n'existe pas.")

    y = df.iloc[:, columns.index(target)].dropna()
    if y.empty:
        raise HTTPException(status_code=400, detail="La colonne cible est entièrement vide.")

    n_unique = int(y.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)

    # --- Détection automatique ---
    if not is_numeric:
        detected = "classification"
        reason = "La cible est catégorielle : classification."
    elif n_unique <= CLASSIFICATION_MAX_UNIQUE:
        detected = "classification"
        reason = f"La cible est numérique mais n'a que {n_unique} valeurs différentes : classification."
    elif (
        _is_integer_like(y)
        and n_unique <= INTEGER_MAX_UNIQUE
        and n_unique / len(y) < INTEGER_RATIO_THRESHOLD
    ):
        detected = "classification"
        reason = f"La cible est entière avec peu de valeurs distinctes ({n_unique}) : classification."
    else:
        detected = "regression"
        reason = "La cible est numérique continue : régression."

    # --- Correction éventuelle par l'utilisateur ---
    problem_type = forced_type or detected
    if forced_type and forced_type != detected:
        reason = f"Type choisi par l'utilisateur ({forced_type}), la détection automatique proposait {detected}."
    elif forced_type:
        reason = f"Type confirmé par l'utilisateur. {reason}"

    if problem_type == "regression" and not is_numeric:
        raise HTTPException(
            status_code=400,
            detail="Une régression nécessite une cible numérique.",
        )

    # --- Vérifications spécifiques à la classification ---
    warnings = []
    n_classes = None

    if problem_type == "classification":
        n_classes = n_unique

        if n_classes < 2:
            raise HTTPException(
                status_code=400,
                detail="La cible ne contient qu'une seule classe : impossible de classifier.",
            )
        if n_classes > MAX_CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"Trop de classes ({n_classes}) pour une classification. Essayez une régression.",
            )

        counts = y.value_counts()
        if counts.min() < 2:
            warnings.append(
                "Certaines classes ont une seule observation : elles seront retirées avant l'entraînement."
            )
        if counts.min() / counts.max() < IMBALANCE_THRESHOLD:
            warnings.append(
                "Classes très déséquilibrées : l'accuracy seule peut être trompeuse, "
                "regardez aussi le recall et le F1-score."
            )

    return {
        "learning_type": "supervised",
        "problem_type": problem_type,
        "target": target,
        "reason": reason,
        "n_classes": n_classes,
        "warnings": warnings,
    }