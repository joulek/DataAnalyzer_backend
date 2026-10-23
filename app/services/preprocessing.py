import re

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Seuils de la partie structurelle (décidée sur le fichier) ---
# Une colonne avec plus de 60 % de valeurs manquantes est ignorée
MISSING_COLUMN_THRESHOLD = 0.6
# Nombre maximal de modalités conservées par colonne catégorielle (le reste -> "Autre")
MAX_CATEGORIES = 30
# Une colonne texte est considérée comme texte libre / identifiant si elle a plus de
# MAX_CATEGORIES valeurs ET si au moins cette part des valeurs est unique
FREE_TEXT_RATIO = 0.5
# Une colonne numérique entière dont cette part des valeurs est unique est un identifiant
ID_UNIQUE_RATIO = 0.95
# Une colonne dont une seule valeur représente cette part (ou plus) est presque constante
CONSTANT_SHARE = 0.99
# Une colonne texte est convertie en nombre si cette part de ses valeurs est numérique
NUMERIC_TEXT_RATIO = 0.95
# Nombre minimal de lignes pour entraîner un modèle
MIN_ROWS = 10

# --- Seuils de la partie apprise (calculée dans le pipeline, sur l'entraînement) ---
# Modalité conservée si elle apparaît dans au moins cette part des lignes
RARE_MIN_FREQUENCY = 0.01
# Coefficient des bornes IQR : Q1 - 1.5*IQR et Q3 + 1.5*IQR
IQR_FACTOR = 1.5
# Asymétrie au-delà de laquelle une variable positive reçoit un log
SKEW_THRESHOLD = 1.0

OTHER_LABEL = "Autre"
ID_NAME_PATTERN = re.compile(r"(^|[_\s])id$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Partie structurelle : décidée sur le fichier, avant le découpage
# ---------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base : noms de colonnes, colonnes vides, doublons."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def _as_number(series: pd.Series) -> pd.Series:
    """Tente de lire une colonne texte comme des nombres (virgule décimale acceptée)."""
    texte = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    texte = texte.where(series.notna())
    return pd.to_numeric(texte, errors="coerce")


def _convert_numeric_text(df: pd.DataFrame, protected: set) -> tuple[pd.DataFrame, list[str]]:
    """Convertit en nombres les colonnes texte qui contiennent surtout des nombres."""
    converted = []
    for col in df.columns:
        if col in protected:
            continue
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_datetime64_any_dtype(serie):
            continue
        valides = serie.notna()
        if not valides.any():
            continue
        nombres = _as_number(serie)
        if nombres[valides].notna().mean() >= NUMERIC_TEXT_RATIO:
            df[col] = nombres
            converted.append(col)
    return df, converted


def _expand_dates(df: pd.DataFrame, protected: set) -> tuple[pd.DataFrame, list[str]]:
    """Remplace chaque colonne de date par son année, son mois et son jour de la semaine."""
    expanded = []
    for col in list(df.columns):
        if col in protected or not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        serie = df[col]
        df[f"{col}_annee"] = serie.dt.year
        df[f"{col}_mois"] = serie.dt.month
        df[f"{col}_jour_semaine"] = serie.dt.dayofweek
        df = df.drop(columns=[col])
        expanded.append(col)
    return df, expanded


def prepare_frame(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Nettoyage structurel complet, sans toucher à la cible."""
    df = clean_dataframe(df)
    protected = {target} if target else set()
    df, _ = _convert_numeric_text(df, protected)
    df, _ = _expand_dates(df, protected)
    return df


def _unique_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    return non_null.nunique() / len(non_null) if len(non_null) else 0.0


def _drop_reason(col: str, series: pd.Series) -> str | None:
    """Renvoie la raison de retrait d'une colonne, ou None si elle est gardée."""
    if series.isna().mean() > MISSING_COLUMN_THRESHOLD:
        return "trop de valeurs manquantes"

    non_null = series.dropna()
    if non_null.nunique() <= 1:
        return "colonne constante"

    if non_null.value_counts(normalize=True).iloc[0] >= CONSTANT_SHARE:
        return "colonne presque constante (une valeur représente plus de 99 %)"

    ratio = _unique_ratio(series)
    nom_identifiant = bool(ID_NAME_PATTERN.search(str(col))) and ratio >= 0.9

    if pd.api.types.is_numeric_dtype(series):
        valeurs = non_null.to_numpy(dtype=float)
        entiere = bool(np.all(np.mod(valeurs, 1) == 0))
        if entiere and ratio >= ID_UNIQUE_RATIO:
            return "identifiant (valeurs toutes différentes)"
        if nom_identifiant:
            return "identifiant (nom de colonne et valeurs uniques)"
    else:
        if non_null.nunique() > MAX_CATEGORIES and ratio >= FREE_TEXT_RATIO:
            return "texte libre ou identifiant"
        if nom_identifiant:
            return "identifiant (nom de colonne et valeurs uniques)"

    return None


def drop_unusable_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Retire les colonnes inutilisables et renvoie la liste retirée avec la raison."""
    kept = []
    dropped = []
    for col in X.columns:
        reason = _drop_reason(col, X[col])
        if reason:
            dropped.append({"column": col, "reason": reason})
        else:
            kept.append(col)
    return X[kept].copy(), dropped


def prepare_supervised(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    """Sépare les variables (X) et la cible (y) pour l'apprentissage supervisé."""
    df = prepare_frame(df, target)

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"La colonne cible '{target}' n'existe pas.")

    # Les lignes sans valeur cible ne peuvent pas servir à l'entraînement
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Pas assez de lignes exploitables (minimum {MIN_ROWS}).",
        )

    y = df[target]
    X, dropped = drop_unusable_columns(df.drop(columns=[target]))

    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="Aucune colonne exploitable pour prédire la cible.",
        )

    return X, y, dropped


def prepare_unsupervised(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Prépare les données pour l'apprentissage non supervisé (pas de cible)."""
    df = prepare_frame(df)

    if len(df) < MIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Pas assez de lignes exploitables (minimum {MIN_ROWS}).",
        )

    X, dropped = drop_unusable_columns(df)

    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Aucune colonne exploitable pour l'analyse.")

    return X, dropped


# ---------------------------------------------------------------------------
# Partie apprise : transformateurs sklearn, ajustés dans le pipeline
# Ils n'apprennent que sur les données qui leur sont passées en fit().
# ---------------------------------------------------------------------------
def _names_out(obj, input_features):
    if input_features is None:
        input_features = [f"x{i}" for i in range(obj.n_features_in_)]
    return np.asarray(input_features, dtype=object)


class SkewLogTransformer(BaseEstimator, TransformerMixin):
    """Applique log(1 + x) aux variables positives et très asymétriques."""

    def __init__(self, threshold: float = SKEW_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.mask_ = np.zeros(X.shape[1], dtype=bool)
        for j in range(X.shape[1]):
            x = X[:, j]
            x = x[np.isfinite(x)]
            if x.size < 3 or x.min() < 0:
                continue
            ecart = x.std()
            if ecart == 0:
                continue
            asymetrie = np.mean(((x - x.mean()) / ecart) ** 3)
            self.mask_[j] = asymetrie > self.threshold
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.mask_] = np.log1p(X[:, self.mask_])
        return X

    def get_feature_names_out(self, input_features=None):
        return _names_out(self, input_features)


class IQRCapper(BaseEstimator, TransformerMixin):
    """Plafonne les valeurs aberrantes aux bornes IQR apprises sur l'entraînement."""

    def __init__(self, factor: float = IQR_FACTOR):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        # Colonne sans dispersion : on ne plafonne pas
        sans_dispersion = iqr == 0
        self.lower_[sans_dispersion] = -np.inf
        self.upper_[sans_dispersion] = np.inf
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return _names_out(self, input_features)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Garde les modalités fréquentes et regroupe les autres dans 'Autre'."""

    def __init__(self, min_frequency: float = RARE_MIN_FREQUENCY, max_categories: int = MAX_CATEGORIES):
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.keep_ = []
        for j in range(X.shape[1]):
            valeurs = pd.Series(X[:, j]).astype(str)
            frequences = valeurs.value_counts(normalize=True)
            frequentes = frequences[frequences >= self.min_frequency].index[: self.max_categories]
            self.keep_.append(set(frequentes))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        out = np.empty(X.shape, dtype=object)
        for j in range(X.shape[1]):
            valeurs = pd.Series(X[:, j]).astype(str).to_numpy()
            garder = self.keep_[j]
            out[:, j] = [v if v in garder else OTHER_LABEL for v in valeurs]
        return out

    def get_feature_names_out(self, input_features=None):
        return _names_out(self, input_features)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le préprocesseur (non entraîné) adapté aux colonnes de X.

    Numérique : médiane -> log si très asymétrique -> plafonnement IQR -> standardisation.
    Catégoriel : valeur la plus fréquente -> regroupement des modalités rares -> One-Hot.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("skew_log", SkewLogTransformer()),
            ("outliers", IQRCapper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper()),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)