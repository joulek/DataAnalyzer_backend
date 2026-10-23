"""Génération d'un notebook Jupyter qui reproduit l'analyse effectuée dans l'application.

Le notebook est autonome : il n'importe pas le package `app`. Les constantes
(graine, seuils, limites) sont lues depuis les services au moment de la génération,
pour rester identiques à celles du backend.
"""

import json
import pprint
import uuid
from datetime import datetime

from app.services.preprocessing import (
    CONSTANT_SHARE,
    FREE_TEXT_RATIO,
    ID_NAME_PATTERN,
    ID_UNIQUE_RATIO,
    IQR_FACTOR,
    MAX_CATEGORIES,
    MIN_ROWS,
    MISSING_COLUMN_THRESHOLD,
    NUMERIC_TEXT_RATIO,
    OTHER_LABEL,
    RARE_MIN_FREQUENCY,
    SKEW_THRESHOLD,
)
from app.services.scoring import SCORE_METRIC_LABEL
from app.services.supervised import MAX_SVM_TRAIN_ROWS, RANDOM_STATE, TEST_SIZE
from app.services.unsupervised import (
    MAX_AGGLOMERATIVE_ROWS,
    MAX_K,
    MAX_MEANSHIFT_ROWS,
    MIN_K,
    SILHOUETTE_SAMPLE_SIZE,
)

PROBLEM_LABELS = {
    "classification": "Classification",
    "regression": "Régression",
    "clustering": "Clustering",
}

ALGO_LABELS = {
    "logistic_regression": "Régression logistique",
    "linear_regression": "Régression linéaire",
    "random_forest": "Random Forest",
    "decision_tree": "Arbre de décision",
    "gradient_boosting": "Gradient Boosting",
    "knn": "K plus proches voisins",
    "svm": "SVM (noyau RBF)",
    "kmeans": "K-Means",
    "agglomerative": "Clustering hiérarchique",
    "dbscan": "DBSCAN",
    "meanshift": "Mean Shift",
    "gmm": "Mélange gaussien (GMM)",
}

# ---------------------------------------------------------------------------
# Code des cellules (texte Python écrit tel quel dans le notebook)
# ---------------------------------------------------------------------------
IMPORTS_CODE = '''
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
'''

LOAD_CODE = '''
def charger(chemin):
    if str(chemin).lower().endswith(".csv"):
        return pd.read_csv(chemin, sep=None, engine="python")
    return pd.read_excel(chemin, engine="openpyxl")


df_brut = charger(CHEMIN_FICHIER)

# Nettoyage de base : noms de colonnes, colonnes vides, doublons (pour l'exploration)
df = df_brut.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.dropna(axis=1, how="all").drop_duplicates().reset_index(drop=True)

print(f"Lignes dans le fichier : {len(df_brut)}")
print(f"Lignes après nettoyage : {len(df)}")
df.head()
'''

EXPLORE_CODE = '''
numerique = df.select_dtypes(include="number")

manquantes = df.isna().sum()
print("Valeurs manquantes par colonne :")
print(manquantes[manquantes > 0] if manquantes.any() else "Aucune valeur manquante")

if not numerique.empty:
    numerique.iloc[:, :12].hist(bins=20, figsize=(12, 8))
    plt.tight_layout()
    plt.show()

if numerique.shape[1] >= 2:
    corr = numerique.iloc[:, :15].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.set_title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()

df.describe(include="all").T
'''

PREP_COMMON_CODE = '''
MOTIF_ID = re.compile(ID_NAME_PATTERN, re.IGNORECASE)


# --- Partie structurelle : décidée sur le fichier ---------------------------
def nettoyer_base(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").drop_duplicates()
    return df.reset_index(drop=True)


def convertir_texte_en_nombres(df, protegees):
    """Convertit en nombres les colonnes texte qui contiennent surtout des nombres."""
    for col in df.columns:
        if col in protegees:
            continue
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_datetime64_any_dtype(serie):
            continue
        valides = serie.notna()
        if not valides.any():
            continue
        texte = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
        nombres = pd.to_numeric(texte.where(valides), errors="coerce")
        if nombres[valides].notna().mean() >= NUMERIC_TEXT_RATIO:
            df[col] = nombres
    return df


def developper_dates(df, protegees):
    """Remplace chaque colonne de date par son année, son mois et son jour de la semaine."""
    for col in list(df.columns):
        if col in protegees or not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        serie = df[col]
        df[f"{col}_annee"] = serie.dt.year
        df[f"{col}_mois"] = serie.dt.month
        df[f"{col}_jour_semaine"] = serie.dt.dayofweek
        df = df.drop(columns=[col])
    return df


def preparer_structure(df, cible=None):
    df = nettoyer_base(df)
    protegees = {cible} if cible else set()
    df = convertir_texte_en_nombres(df, protegees)
    df = developper_dates(df, protegees)
    return df


def raison_retrait(col, serie):
    """Renvoie la raison de retrait d'une colonne, ou None si elle est gardée."""
    if serie.isna().mean() > MISSING_COLUMN_THRESHOLD:
        return "trop de valeurs manquantes"

    non_nulles = serie.dropna()
    if non_nulles.nunique() <= 1:
        return "colonne constante"
    if non_nulles.value_counts(normalize=True).iloc[0] >= CONSTANT_SHARE:
        return "colonne presque constante (une valeur représente plus de 99 %)"

    ratio = non_nulles.nunique() / len(non_nulles)
    nom_identifiant = bool(MOTIF_ID.search(str(col))) and ratio >= 0.9

    if pd.api.types.is_numeric_dtype(serie):
        valeurs = non_nulles.to_numpy(dtype=float)
        entiere = bool(np.all(np.mod(valeurs, 1) == 0))
        if entiere and ratio >= ID_UNIQUE_RATIO:
            return "identifiant (valeurs toutes différentes)"
        if nom_identifiant:
            return "identifiant (nom de colonne et valeurs uniques)"
    else:
        if non_nulles.nunique() > MAX_CATEGORIES and ratio >= FREE_TEXT_RATIO:
            return "texte libre ou identifiant"
        if nom_identifiant:
            return "identifiant (nom de colonne et valeurs uniques)"
    return None


def retirer_colonnes_inutiles(X):
    """Retire les colonnes inutilisables et renvoie la liste avec la raison."""
    garder, retirees = [], []
    for col in X.columns:
        raison = raison_retrait(col, X[col])
        if raison:
            retirees.append({"column": col, "reason": raison})
        else:
            garder.append(col)
    return X[garder].copy(), retirees


def preparer_supervise(df_brut, cible):
    df_p = preparer_structure(df_brut, cible)
    if cible not in df_p.columns:
        raise ValueError(f"La colonne cible '{cible}' n'existe pas.")
    df_p = df_p.dropna(subset=[cible]).reset_index(drop=True)
    if len(df_p) < MIN_ROWS:
        raise ValueError(f"Pas assez de lignes exploitables (minimum {MIN_ROWS}).")
    y = df_p[cible]
    X, retirees = retirer_colonnes_inutiles(df_p.drop(columns=[cible]))
    if X.shape[1] == 0:
        raise ValueError("Aucune colonne exploitable pour prédire la cible.")
    return X, y, retirees


def preparer_non_supervise(df_brut):
    df_p = preparer_structure(df_brut)
    if len(df_p) < MIN_ROWS:
        raise ValueError(f"Pas assez de lignes exploitables (minimum {MIN_ROWS}).")
    X, retirees = retirer_colonnes_inutiles(df_p)
    if X.shape[1] == 0:
        raise ValueError("Aucune colonne exploitable pour l'analyse.")
    return X, retirees


# --- Partie apprise : ajustée dans le pipeline, donc sur l'entraînement uniquement ---
class TransformationAsymetrie(BaseEstimator, TransformerMixin):
    """Applique log(1 + x) aux variables positives et très asymétriques."""

    def __init__(self, threshold=SKEW_THRESHOLD):
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


class PlafonnementIQR(BaseEstimator, TransformerMixin):
    """Plafonne les valeurs aberrantes aux bornes IQR apprises sur l'entraînement."""

    def __init__(self, factor=IQR_FACTOR):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        sans_dispersion = iqr == 0
        self.lower_[sans_dispersion] = -np.inf
        self.upper_[sans_dispersion] = np.inf
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class RegroupementModalites(BaseEstimator, TransformerMixin):
    """Garde les modalités fréquentes et regroupe les autres dans 'Autre'."""

    def __init__(self, min_frequency=RARE_MIN_FREQUENCY, max_categories=MAX_CATEGORIES):
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


def build_preprocessor(X):
    """Numérique : médiane -> log si très asymétrique -> plafonnement IQR -> standardisation.
    Catégoriel : valeur la plus fréquente -> regroupement des modalités rares -> One-Hot."""
    colonnes_numeriques = X.select_dtypes(include="number").columns.tolist()
    colonnes_categorielles = [c for c in X.columns if c not in colonnes_numeriques]

    transformateurs = []
    if colonnes_numeriques:
        transformateurs.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("skew_log", TransformationAsymetrie()),
                ("outliers", PlafonnementIQR()),
                ("scaler", StandardScaler()),
            ]),
            colonnes_numeriques,
        ))
    if colonnes_categorielles:
        transformateurs.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("rare", RegroupementModalites()),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            colonnes_categorielles,
        ))
    return ColumnTransformer(transformers=transformateurs)
'''

SUPERVISED_PREP_CODE = '''
TARGET = CONFIG["target"]
PROBLEME = CONFIG["problem_type"]

X, y, colonnes_retirees = preparer_supervise(df_brut, TARGET)

if PROBLEME == "classification":
    # Les classes avec une seule observation ne peuvent pas être stratifiées
    comptes = y.value_counts()
    valides = comptes[comptes >= 2].index
    masque = y.isin(valides)
    X, y = X[masque], y[masque]
    if y.nunique() < 2:
        raise ValueError("Il reste moins de 2 classes exploitables après nettoyage.")
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str)
    encodeur = LabelEncoder()
    y_valeurs = encodeur.fit_transform(y)
    noms_classes = [str(c) for c in encodeur.classes_]
else:
    y_valeurs = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    noms_classes = None

stratify = None
if PROBLEME == "classification":
    n_test = int(np.ceil(len(y_valeurs) * TEST_SIZE))
    if n_test >= len(noms_classes):
        stratify = y_valeurs

X_train, X_test, y_train, y_test = train_test_split(
    X, y_valeurs, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
)
print(f"Entraînement : {len(X_train)} lignes | Test : {len(X_test)} lignes")
pd.DataFrame(colonnes_retirees)
'''

CATALOGUE_CODE = '''
CATALOGUE_CLASSIFICATION = {
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

CATALOGUE_REGRESSION = {
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

CATALOGUE = CATALOGUE_CLASSIFICATION if PROBLEME == "classification" else CATALOGUE_REGRESSION

# Algorithmes limités en lignes d'entraînement (comme dans l'application)
LIMITES_LIGNES = {"svm": MAX_SVM_TRAIN_ROWS}
'''

SUPERVISED_TRAIN_CODE = '''
# Entraînement : le préprocesseur est dans le pipeline, donc entraîné sur X_train seulement
resultats = {}
for cle in CONFIG["algorithms"]:
    nom, fabrique = CATALOGUE[cle]
    if cle in LIMITES_LIGNES and len(X_train) > LIMITES_LIGNES[cle]:
        print(f"{nom} ignoré : dataset trop grand (max {LIMITES_LIGNES[cle]} lignes d'entraînement).")
        continue
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X)),
        ("model", fabrique()),
    ])
    pipeline.fit(X_train, y_train)
    resultats[cle] = {"nom": nom, "pipeline": pipeline, "y_pred": pipeline.predict(X_test)}

lignes = []
for cle, r in resultats.items():
    if PROBLEME == "classification":
        labels = list(range(len(noms_classes)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, r["y_pred"], labels=labels, average="weighted", zero_division=0
        )
        brut = f1
        ligne = {
            "Modèle": r["nom"],
            "Accuracy": accuracy_score(y_test, r["y_pred"]),
            "Precision": precision,
            "Recall": recall,
            "F1 pondéré": f1,
        }
    else:
        brut = r2_score(y_test, r["y_pred"])
        ligne = {
            "Modèle": r["nom"],
            "R²": brut,
            "MAE": mean_absolute_error(y_test, r["y_pred"]),
            "RMSE": np.sqrt(mean_squared_error(y_test, r["y_pred"])),
        }
    # Score : métrique principale ramenée entre 0 et 100 %
    ligne["Score (%)"] = round(max(0.0, min(1.0, float(brut))) * 100, 1)
    ligne["_cle"] = cle
    ligne["_brut"] = float(brut)
    lignes.append(ligne)

meilleur_cle = max(lignes, key=lambda l: l["_brut"])["_cle"]
tableau = pd.DataFrame(
    [{k: v for k, v in l.items() if not k.startswith("_")} for l in lignes]
).round(4)
tableau
'''

SUPERVISED_EVAL_CODE = '''
meilleur = resultats[meilleur_cle]
print("Meilleur modèle :", meilleur["nom"])

fig, ax = plt.subplots(figsize=(6, 5))
if PROBLEME == "classification":
    labels = list(range(len(noms_classes)))
    matrice = confusion_matrix(y_test, meilleur["y_pred"], labels=labels)
    ax.imshow(matrice, cmap="Blues")
    ax.set_xticks(labels, noms_classes)
    ax.set_yticks(labels, noms_classes)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    for i in labels:
        for j in labels:
            ax.text(j, i, matrice[i, j], ha="center", va="center")
    ax.set_title("Matrice de confusion (meilleur modèle)")
else:
    ax.scatter(y_test, meilleur["y_pred"], alpha=0.6)
    borne = [
        min(np.min(y_test), np.min(meilleur["y_pred"])),
        max(np.max(y_test), np.max(meilleur["y_pred"])),
    ]
    ax.plot(borne, borne, "--", color="red")
    ax.set_xlabel("Valeur réelle")
    ax.set_ylabel("Valeur prédite")
    ax.set_title("Valeurs réelles vs prédites (meilleur modèle)")
plt.tight_layout()
plt.show()
'''

UNSUPERVISED_PREP_CODE = '''
X, colonnes_retirees = preparer_non_supervise(df_brut)

donnees = build_preprocessor(X).fit_transform(X)
print("Données prétraitées :", donnees.shape)
pd.DataFrame(colonnes_retirees)
'''

K_SELECTION_CODE = '''
def silhouette(data, labels):
    taille = SILHOUETTE_SAMPLE_SIZE if len(data) > SILHOUETTE_SAMPLE_SIZE else None
    return float(silhouette_score(data, labels, sample_size=taille, random_state=RANDOM_STATE))


max_k = min(MAX_K, len(donnees) - 1)

if CONFIG["n_clusters"] is not None:
    k = CONFIG["n_clusters"]
    print("Nombre de clusters imposé :", k)
else:
    historique = []
    for k_test in range(MIN_K, max_k + 1):
        modele_k = KMeans(n_clusters=k_test, n_init=10, random_state=RANDOM_STATE).fit(donnees)
        if len(set(modele_k.labels_)) < 2:
            continue
        historique.append({"k": k_test, "silhouette": silhouette(donnees, modele_k.labels_)})
    if historique:
        k = max(historique, key=lambda h: h["silhouette"])["k"]
        print("k choisi automatiquement (silhouette K-Means) :", k)
        pd.DataFrame(historique)
    else:
        # Aucun k exploitable : seuls DBSCAN et Mean Shift décident eux-mêmes du nombre de clusters
        k = MIN_K
'''

UNSUPERVISED_TRAIN_CODE = '''
NOMS = {
    "kmeans": "K-Means",
    "agglomerative": "Clustering hiérarchique",
    "dbscan": "DBSCAN",
    "meanshift": "Mean Shift",
    "gmm": "Mélange gaussien (GMM)",
}


def estimer_dbscan(data):
    """eps estimé à partir de la distance au k-ième voisin, comme dans l'application."""
    min_samples = int(min(max(5, 2 * data.shape[1]), 15, len(data) - 1))
    voisins = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, _ = voisins.kneighbors(data)
    eps = max(float(np.percentile(distances[:, -1], 90)), 1e-6)
    return eps, min_samples


def ajuster(cle):
    """Renvoie les labels, ou None si l'algorithme ne peut pas être appliqué."""
    if cle == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(donnees).labels_
    if cle == "agglomerative":
        if len(donnees) > MAX_AGGLOMERATIVE_ROWS:
            return None
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(donnees)
    if cle == "gmm":
        return GaussianMixture(
            n_components=k, covariance_type="full", random_state=RANDOM_STATE
        ).fit(donnees).predict(donnees)
    if cle == "meanshift":
        if len(donnees) > MAX_MEANSHIFT_ROWS:
            return None
        largeur = estimate_bandwidth(
            donnees, quantile=0.2, n_samples=min(len(donnees), 500), random_state=RANDOM_STATE
        )
        if not largeur or largeur <= 0:
            return None
        return MeanShift(bandwidth=largeur, bin_seeding=True).fit_predict(donnees)
    # DBSCAN
    eps, min_samples = estimer_dbscan(donnees)
    print(f"DBSCAN : eps = {eps:.4f}, min_samples = {min_samples}")
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(donnees)


def metriques(labels):
    """Métriques internes, sans les points de bruit (label -1)."""
    masque = labels != -1
    d, l = donnees[masque], labels[masque]
    if len(set(l)) < 2 or len(set(l)) > len(d) - 1:
        return None
    return {
        "silhouette": silhouette(d, l),
        "davies_bouldin": float(davies_bouldin_score(d, l)),
        "calinski_harabasz": float(calinski_harabasz_score(d, l)),
    }


labels_par_algo = {}
lignes = []
for cle in CONFIG["algorithms"]:
    labels = ajuster(cle)
    if labels is None:
        print(f"{NOMS[cle]} ignoré : dataset trop grand ou algorithme non applicable.")
        continue
    m = metriques(labels)
    if m is None:
        print(f"{NOMS[cle]} : moins de 2 clusters exploitables.")
        continue
    labels_par_algo[cle] = labels
    lignes.append({
        "Algorithme": NOMS[cle],
        "Clusters": int(len(set(labels) - {-1})),
        "Bruit": int((labels == -1).sum()),
        "Silhouette": m["silhouette"],
        "Davies-Bouldin": m["davies_bouldin"],
        "Calinski-Harabasz": m["calinski_harabasz"],
        "Score (%)": round(max(0.0, min(1.0, m["silhouette"])) * 100, 1),
        "_cle": cle,
        "_brut": m["silhouette"],
    })

meilleur_cle = max(lignes, key=lambda l: l["_brut"])["_cle"]
tableau = pd.DataFrame(
    [{k_: v for k_, v in l.items() if not k_.startswith("_")} for l in lignes]
).round(4)
tableau
'''

UNSUPERVISED_EVAL_CODE = '''
meilleur_labels = labels_par_algo[meilleur_cle]

if donnees.shape[1] >= 2:
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(donnees)
else:
    coords = np.column_stack([donnees[:, 0], np.zeros(len(donnees))])

fig, ax = plt.subplots(figsize=(7, 5))
for valeur in sorted(set(meilleur_labels)):
    masque = meilleur_labels == valeur
    nom = "Bruit" if valeur == -1 else f"Cluster {valeur}"
    ax.scatter(coords[masque, 0], coords[masque, 1], s=15, alpha=0.7, label=nom)
ax.set_title(f"Clusters (projection PCA) : {NOMS[meilleur_cle]}")
ax.set_xlabel("Axe 1")
ax.set_ylabel("Axe 2")
ax.legend()
plt.tight_layout()
plt.show()
'''

FINAL_CODE = '''
print("Meilleur modèle (notebook) :", meilleur_cle)
score_notebook = max(l["Score (%)"] for l in lignes if l["_cle"] == meilleur_cle)
print("Score (notebook)    :", score_notebook, "%")
print("Score (application) :", CONFIG["best_score"], "%")
'''


# ---------------------------------------------------------------------------
# Outils de construction du notebook (format nbformat 4)
# ---------------------------------------------------------------------------
def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source(text),
    }


def _build_config(analysis: dict, filename: str) -> dict:
    detection = analysis["detection"]
    results = analysis["results"]
    learning = detection["learning_type"]
    models = results.get("models", [])
    best = next((m for m in models if m["key"] == results.get("best_model")), {})

    return {
        "filename": filename,
        "learning_type": learning,
        "problem_type": results["problem_type"],
        "target": detection.get("target"),
        "algorithms": [m["key"] for m in models],
        # Nombre de clusters imposé seulement si l'utilisateur l'a choisi
        "n_clusters": None if (learning == "supervised" or results.get("auto_k")) else results.get("k"),
        "best_model": results.get("best_model"),
        "best_name": best.get("name"),
        "best_score": results.get("best_score"),
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


def _config_code(config: dict) -> str:
    return (
        "CONFIG = " + pprint.pformat(config, sort_dicts=False, width=88) + "\n\n"
        f"CHEMIN_FICHIER = {config['filename']!r}  # modifiez ce chemin si besoin\n\n"
        "# Seuils du prétraitement (identiques à l'application)\n"
        f"MISSING_COLUMN_THRESHOLD = {MISSING_COLUMN_THRESHOLD!r}\n"
        f"MAX_CATEGORIES = {MAX_CATEGORIES!r}\n"
        f"MIN_ROWS = {MIN_ROWS!r}\n"
        f"FREE_TEXT_RATIO = {FREE_TEXT_RATIO!r}\n"
        f"ID_UNIQUE_RATIO = {ID_UNIQUE_RATIO!r}\n"
        f"CONSTANT_SHARE = {CONSTANT_SHARE!r}\n"
        f"NUMERIC_TEXT_RATIO = {NUMERIC_TEXT_RATIO!r}\n"
        f"ID_NAME_PATTERN = {ID_NAME_PATTERN.pattern!r}\n"
        f"RARE_MIN_FREQUENCY = {RARE_MIN_FREQUENCY!r}\n"
        f"IQR_FACTOR = {IQR_FACTOR!r}\n"
        f"SKEW_THRESHOLD = {SKEW_THRESHOLD!r}\n"
        f"OTHER_LABEL = {OTHER_LABEL!r}\n\n"
        "# Entraînement et clustering (identiques à l'application)\n"
        f"RANDOM_STATE = {RANDOM_STATE!r}\n"
        f"TEST_SIZE = {TEST_SIZE!r}\n"
        f"MAX_SVM_TRAIN_ROWS = {MAX_SVM_TRAIN_ROWS!r}\n"
        f"MIN_K = {MIN_K!r}\n"
        f"MAX_K = {MAX_K!r}\n"
        f"SILHOUETTE_SAMPLE_SIZE = {SILHOUETTE_SAMPLE_SIZE!r}\n"
        f"MAX_AGGLOMERATIVE_ROWS = {MAX_AGGLOMERATIVE_ROWS!r}\n"
        f"MAX_MEANSHIFT_ROWS = {MAX_MEANSHIFT_ROWS!r}\n"
    )


def _intro(config: dict) -> dict:
    probleme = PROBLEM_LABELS.get(config["problem_type"], config["problem_type"])
    if config["learning_type"] == "supervised":
        cible = f"Colonne cible : **{config['target']}**"
    else:
        cible = "Aucune colonne cible : analyse non supervisée (clustering)"
    algos = ", ".join(ALGO_LABELS.get(a, a) for a in config["algorithms"])
    return _md(
        f"""# Analyse : {config['filename']}

Ce notebook reproduit l'analyse réalisée dans DataAnalyzer, étape par étape.

- Type de problème : **{probleme}**
- {cible}
- Algorithmes : {algos}
- Généré le {config['generated_at']}

Pour l'exécuter, il faut `pandas`, `scikit-learn`, `matplotlib` et `openpyxl` (pour un fichier `.xlsx`)."""
    )


def _conclusion(config: dict) -> dict:
    base = SCORE_METRIC_LABEL.get(config["problem_type"], "")
    return _md(
        f"""## 10. Conclusion

Le meilleur modèle est **{config['best_name']}**, avec un score de **{config['best_score']} %**.
Ce score est basé sur le {base}. Il est ramené entre 0 et 100 % : une valeur négative devient 0 %."""
    )


def _cells(config: dict) -> list[dict]:
    supervised = config["learning_type"] == "supervised"

    cells = [
        _intro(config),
        _md("## 1. Bibliothèques"),
        _code(IMPORTS_CODE),
        _md(
            "## 2. Paramètres\n\nCes valeurs viennent de l'analyse effectuée dans l'application. "
            "Modifiez `CHEMIN_FICHIER` si votre fichier n'est pas dans le même dossier que ce notebook."
        ),
        _code(_config_code(config)),
        _md("## 3. Chargement et nettoyage"),
        _code(LOAD_CODE),
        _md("## 4. Exploration\n\nDistribution, corrélations et valeurs manquantes."),
        _code(EXPLORE_CODE),
        _md(
            "## 5. Préparation des données\n\n"
            "Étapes : suppression des identifiants et des colonnes inutilisables, conversion des "
            "nombres écrits en texte, extraction année / mois / jour pour les dates, regroupement "
            "des modalités rares dans « Autre », passage en logarithme des variables très "
            "asymétriques, plafonnement des valeurs aberrantes (méthode IQR), standardisation, "
            "puis encodage One-Hot. Les transformations apprises sont calculées sur l'entraînement "
            "uniquement (supervisé) pour éviter toute fuite de données."
        ),
        _code(PREP_COMMON_CODE),
    ]

    if supervised:
        cells += [
            _md(
                "## 6. Cible et découpage train / test\n\n"
                f"Les données sont découpées : {TEST_SIZE:.0%} pour le test, graine aléatoire {RANDOM_STATE}."
            ),
            _code(SUPERVISED_PREP_CODE),
            _md("## 7. Entraînement des modèles"),
            _code(CATALOGUE_CODE),
            _code(SUPERVISED_TRAIN_CODE),
            _md("## 8. Évaluation du meilleur modèle"),
            _code(SUPERVISED_EVAL_CODE),
        ]
    else:
        cells += [
            _md("## 6. Préparation du clustering"),
            _code(UNSUPERVISED_PREP_CODE),
            _md(
                "## 7. Choix du nombre de clusters\n\n"
                "Si aucun k n'a été imposé, le meilleur est choisi avec la silhouette de K-Means."
            ),
            _code(K_SELECTION_CODE),
            _md("## 8. Entraînement des algorithmes"),
            _code(UNSUPERVISED_TRAIN_CODE),
            _md("## 9. Visualisation des clusters"),
            _code(UNSUPERVISED_EVAL_CODE),
        ]

    cells += [_conclusion(config), _code(FINAL_CODE)]
    return cells


def build_notebook(analysis: dict, filename: str) -> dict:
    """Construit le notebook (structure nbformat 4) à partir de la réponse de /analyze."""
    config = _build_config(analysis, filename)
    return {
        "cells": _cells(config),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook_bytes(analysis: dict, filename: str) -> bytes:
    """Renvoie le notebook sous forme de fichier .ipynb (JSON encodé en UTF-8)."""
    notebook = build_notebook(analysis, filename)
    return json.dumps(notebook, ensure_ascii=False, indent=1).encode("utf-8")