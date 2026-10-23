import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from app.services.preprocessing import build_preprocessor
from app.services.scoring import compute_score

RANDOM_STATE = 42
MIN_K = 2
MAX_K = 8
# Le calcul de la silhouette est coûteux : on l'estime sur un échantillon
SILHOUETTE_SAMPLE_SIZE = 5000
# Nombre maximal de points renvoyés pour le nuage de points
MAX_POINTS_RETURNED = 1500
# Nombre maximal de colonnes affichées dans le profil des clusters
MAX_PROFILE_COLUMNS = 10
# Le clustering hiérarchique consomme beaucoup de mémoire (au carré du nombre de lignes)
MAX_AGGLOMERATIVE_ROWS = 5000
# Mean Shift calcule les distances entre tous les points : limité comme le hiérarchique
MAX_MEANSHIFT_ROWS = 5000

ALGORITHMS = {
    "kmeans": "K-Means",
    "agglomerative": "Clustering hiérarchique",
    "dbscan": "DBSCAN",
    "meanshift": "Mean Shift",
    "gmm": "Mélange gaussien (GMM)",
}


# ---------------------------------------------------------------------------
# Outils communs
# ---------------------------------------------------------------------------
def _silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette (estimée sur un échantillon si le dataset est grand)."""
    sample_size = SILHOUETTE_SAMPLE_SIZE if len(data) > SILHOUETTE_SAMPLE_SIZE else None
    return float(
        silhouette_score(data, labels, sample_size=sample_size, random_state=RANDOM_STATE)
    )


def _fit_kmeans(data: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(data)


def _select_k(data: np.ndarray, max_k: int) -> list[dict]:
    """Teste plusieurs valeurs de k (avec K-Means) : inertie + silhouette pour chacune."""
    scores = []
    for k in range(MIN_K, max_k + 1):
        model = _fit_kmeans(data, k)
        if len(set(model.labels_)) < 2:
            continue
        scores.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 2),
                "silhouette": round(_silhouette(data, model.labels_), 4),
            }
        )
    return scores


def _clustering_metrics(data: np.ndarray, labels: np.ndarray) -> dict:
    """Métriques internes. Les points de bruit (label -1) sont exclus du calcul."""
    mask = labels != -1
    clean_data, clean_labels = data[mask], labels[mask]

    n_found = len(set(clean_labels))
    if n_found < 2 or n_found > len(clean_data) - 1:
        raise ValueError("moins de 2 clusters exploitables")

    return {
        "silhouette": round(_silhouette(clean_data, clean_labels), 4),
        "davies_bouldin": round(float(davies_bouldin_score(clean_data, clean_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(clean_data, clean_labels)), 2),
    }


def _cluster_profiles(X: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Taille de chaque cluster et moyenne des variables numériques."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    # On garde les colonnes numériques les plus variables (en valeur relative)
    if len(numeric_cols) > MAX_PROFILE_COLUMNS:
        spread = (X[numeric_cols].std() / (X[numeric_cols].mean().abs() + 1e-9)).abs()
        numeric_cols = spread.sort_values(ascending=False).index[:MAX_PROFILE_COLUMNS].tolist()

    profiles = []
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        means = X.loc[mask, numeric_cols].mean() if numeric_cols else {}
        profiles.append(
            {
                "cluster": int(cluster),
                "is_noise": bool(cluster == -1),
                "size": int(mask.sum()),
                "percentage": round(float(mask.mean() * 100), 1),
                "means": {col: round(float(means[col]), 3) for col in numeric_cols},
            }
        )
    return profiles


# ---------------------------------------------------------------------------
# Projection 2D (calculée une seule fois, réutilisée par tous les algorithmes)
# ---------------------------------------------------------------------------
def _compute_projection(data: np.ndarray) -> dict:
    """PCA en 2 dimensions + échantillon de points à afficher."""
    n_points = len(data)
    rng = np.random.default_rng(RANDOM_STATE)
    index = (
        rng.choice(n_points, MAX_POINTS_RETURNED, replace=False)
        if n_points > MAX_POINTS_RETURNED
        else np.arange(n_points)
    )

    if data.shape[1] < 2:
        coords = np.column_stack([data[:, 0], np.zeros(n_points)])
        explained = [1.0, 0.0]
    else:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(data)
        explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    return {"coords": coords, "index": index, "explained_variance": explained}


def _projection_payload(projection: dict, labels: np.ndarray) -> dict:
    """Format JSON du nuage de points pour un algorithme donné."""
    index, coords = projection["index"], projection["coords"]
    return {
        "x": [round(float(v), 4) for v in coords[index, 0]],
        "y": [round(float(v), 4) for v in coords[index, 1]],
        "labels": [int(v) for v in labels[index]],
        "explained_variance": projection["explained_variance"],
    }


# ---------------------------------------------------------------------------
# Les algorithmes
# ---------------------------------------------------------------------------
def _estimate_dbscan_params(data: np.ndarray) -> tuple[float, int]:
    """Estime eps avec la méthode de la distance au k-ième voisin."""
    min_samples = int(min(max(5, 2 * data.shape[1]), 15, len(data) - 1))
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, _ = neighbors.kneighbors(data)
    eps = max(float(np.percentile(distances[:, -1], 90)), 1e-6)
    return eps, min_samples


def _fit_algorithm(key: str, data: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """Entraîne un algorithme et renvoie (labels, paramètres utilisés)."""
    if key == "kmeans":
        model = _fit_kmeans(data, k)
        return model.labels_, {"n_clusters": k, "inertia": round(float(model.inertia_), 2)}

    if key == "agglomerative":
        if len(data) > MAX_AGGLOMERATIVE_ROWS:
            raise ValueError(
                f"dataset trop grand pour le clustering hiérarchique (max {MAX_AGGLOMERATIVE_ROWS} lignes)"
            )
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data)
        return labels, {"n_clusters": k, "linkage": "ward"}

    if key == "meanshift":
        if len(data) > MAX_MEANSHIFT_ROWS:
            raise ValueError(
                f"dataset trop grand pour Mean Shift (max {MAX_MEANSHIFT_ROWS} lignes)"
            )
        # La largeur de bande est estimée à partir des données : pas de k à fixer
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=min(len(data), 500), random_state=RANDOM_STATE)
        if not bandwidth or bandwidth <= 0:
            raise ValueError("largeur de bande nulle : données trop homogènes")
        labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(data)
        return labels, {"bandwidth": round(float(bandwidth), 4)}

    if key == "gmm":
        # Mélange gaussien : chaque cluster est une gaussienne, les points ont une appartenance souple
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
        labels = model.fit(data).predict(data)
        return labels, {"n_components": k, "covariance_type": "full"}

    # DBSCAN
    eps, min_samples = _estimate_dbscan_params(data)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
    return labels, {"eps": round(eps, 4), "min_samples": min_samples}


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------
def run_unsupervised(
    X: pd.DataFrame,
    n_clusters: int | None = None,
    algorithms: list[str] | None = None,
) -> dict:
    """Clustering avec plusieurs algorithmes, comparés par la silhouette.

    n_clusters : nombre de clusters (K-Means, hiérarchique et GMM). Si None, le meilleur
                 k (2 à 8) est choisi avec la silhouette de K-Means.
    algorithms : liste de clés parmi "kmeans", "agglomerative", "dbscan", "meanshift", "gmm"
                 (par défaut, tous).
    """
    X = X.reset_index(drop=True)

    selected = algorithms or list(ALGORITHMS.keys())
    unknown = [key for key in selected if key not in ALGORITHMS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithme(s) inconnu(s) : {', '.join(unknown)}. "
            f"Disponibles : {', '.join(ALGORITHMS.keys())}",
        )

    max_k = min(MAX_K, len(X) - 1)
    if max_k < MIN_K:
        raise HTTPException(status_code=400, detail="Pas assez de lignes pour faire du clustering.")

    if n_clusters is not None and not (MIN_K <= n_clusters <= len(X) - 1):
        raise HTTPException(
            status_code=400,
            detail=f"Le nombre de clusters doit être compris entre {MIN_K} et {len(X) - 1}.",
        )

    # Le préprocesseur est entraîné sur tout le dataset (pas de jeu de test ici)
    data = build_preprocessor(X).fit_transform(X)

    # --- Choix de k (partagé par K-Means, hiérarchique et GMM) ---
    k_selection = _select_k(data, max_k)
    if n_clusters is not None:
        k, auto_k = n_clusters, False
    elif k_selection:
        k = max(k_selection, key=lambda item: item["silhouette"])["k"]
        auto_k = True
    elif "dbscan" in selected or "meanshift" in selected:
        k, auto_k = MIN_K, True  # sera ignoré par DBSCAN et Mean Shift
    else:
        raise HTTPException(
            status_code=400,
            detail="Impossible de former des clusters distincts avec ces données.",
        )

    projection = _compute_projection(data)

    # --- Entraînement de chaque algorithme ---
    results = []
    errors = []

    for key in selected:
        label = ALGORITHMS[key]
        try:
            labels, params = _fit_algorithm(key, data, k)
            metrics = _clustering_metrics(data, labels)
        except Exception as exc:
            errors.append({"model": label, "error": str(exc)})
            continue

        n_noise = int((labels == -1).sum())
        results.append(
            {
                "key": key,
                "name": label,
                "n_clusters": int(len(set(labels) - {-1})),
                "n_noise": n_noise,
                "params": params,
                "metrics": metrics,
                "score": compute_score("clustering", metrics),
                "profiles": _cluster_profiles(X, labels),
                "projection": _projection_payload(projection, labels),
            }
        )

    if not results:
        raise HTTPException(
            status_code=400,
            detail=f"Aucun algorithme n'a pu former de clusters exploitables. Détails : {errors}",
        )

    best = max(results, key=lambda r: r["metrics"]["silhouette"])

    return {
        "problem_type": "clustering",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "k": int(k),
        "auto_k": auto_k,
        "k_selection": k_selection,
        "models": results,
        "failed_models": errors,
        "best_model": best["key"],
        "best_metric": "silhouette",
        "best_score": best["score"],
    }