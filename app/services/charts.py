import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

TEMPLATE = "plotly_white"
PALETTE = qualitative.Plotly
NOISE_COLOR = "#9e9e9e"

MAX_HISTOGRAMS = 12
HIST_BINS = 20
MAX_HEATMAP_COLUMNS = 15


# ---------------------------------------------------------------------------
# Outils communs
# ---------------------------------------------------------------------------
def _to_dict(
    fig: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 400,
) -> dict:
    """Applique le style commun et convertit la figure en dictionnaire JSON."""
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    # to_json gère numpy et NaN ; json.loads donne un dict que FastAPI sait renvoyer
    return json.loads(fig.to_json())


def _cluster_color(cluster: int) -> str:
    return NOISE_COLOR if cluster == -1 else PALETTE[cluster % len(PALETTE)]


def _cluster_name(cluster: int) -> str:
    return "Bruit" if cluster == -1 else f"Cluster {cluster}"


def _values(models: list[dict], getter) -> list:
    """Extrait une valeur par modèle. Les valeurs absentes deviennent None (barre vide)."""
    out = []
    for m in models:
        try:
            out.append(getter(m))
        except (KeyError, TypeError):
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Graphiques du dataset (affichés avant ou pendant l'analyse)
# ---------------------------------------------------------------------------
def _histogram(series: pd.Series, column: str) -> dict:
    """Histogramme calculé côté serveur : seules les barres sont envoyées au frontend."""
    counts, edges = np.histogram(series, bins=HIST_BINS)
    centers = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure(
        go.Bar(
            x=np.round(centers, 4).tolist(),
            y=counts.tolist(),
            width=np.diff(edges).tolist(),
            marker_color=PALETTE[0],
        )
    )
    fig.update_layout(bargap=0, showlegend=False)
    return _to_dict(fig, f"Distribution de {column}", column, "Effectif", height=320)


def _correlation_heatmap(df: pd.DataFrame) -> dict | None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None

    # On garde les colonnes les plus variables si elles sont trop nombreuses
    if numeric.shape[1] > MAX_HEATMAP_COLUMNS:
        keep = numeric.var().sort_values(ascending=False).index[:MAX_HEATMAP_COLUMNS]
        numeric = numeric[keep]

    corr = numeric.corr().round(2)
    z = [[None if np.isnan(v) else float(v) for v in row] for row in corr.to_numpy()]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=z,
            texttemplate="%{text}",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _to_dict(fig, "Matrice de corrélation", height=500)


def _missing_values_chart(df: pd.DataFrame) -> dict | None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    percentages = (missing / len(df) * 100).round(1)
    fig = go.Figure(
        go.Bar(
            x=[str(c) for c in percentages.index],
            y=percentages.tolist(),
            marker_color=PALETTE[1],
        )
    )
    return _to_dict(fig, "Valeurs manquantes par colonne", "Colonne", "% de valeurs manquantes")


def build_dataset_charts(df: pd.DataFrame) -> dict:
    """Histogrammes, corrélations et valeurs manquantes d'un dataset."""
    histograms = []
    for column in df.select_dtypes(include="number").columns[:MAX_HISTOGRAMS]:
        series = df[column].dropna()
        if series.empty:
            continue
        histograms.append({"column": str(column), "figure": _histogram(series, str(column))})

    return {
        "histograms": histograms,
        "correlation": _correlation_heatmap(df),
        "missing_values": _missing_values_chart(df),
    }


# ---------------------------------------------------------------------------
# Supervisé
# ---------------------------------------------------------------------------
def _feature_importance_chart(items: list[dict]) -> dict | None:
    if not items:
        return None
    # Ordre inversé : la variable la plus importante s'affiche en haut
    names = [item["feature"] for item in items][::-1]
    values = [item["importance"] for item in items][::-1]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=PALETTE[2]))
    return _to_dict(fig, "Importance des variables", "Importance", height=max(300, 30 * len(items) + 120))


def _confusion_matrix_chart(cm: dict) -> dict:
    labels = cm["labels"]
    matrix = cm["matrix"]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    # type="category" : évite que des classes "0", "1", "2" soient lues comme des nombres
    fig.update_xaxes(type="category", title_text="Classe prédite")
    fig.update_yaxes(type="category", title_text="Classe réelle", autorange="reversed")
    return _to_dict(fig, "Matrice de confusion", height=450)


def _per_class_chart(per_class: list[dict]) -> dict:
    classes = [item["class"] for item in per_class]
    fig = go.Figure()
    for metric, label, color in [
        ("precision", "Precision", PALETTE[0]),
        ("recall", "Recall", PALETTE[1]),
        ("f1", "F1-score", PALETTE[2]),
    ]:
        fig.add_trace(
            go.Bar(name=label, x=classes, y=[item[metric] for item in per_class], marker_color=color)
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, 1.05])
    return _to_dict(fig, "Precision, recall et F1 par classe", "Classe", "Score")


def _roc_chart(roc: dict) -> dict:
    """Courbe ROC : une courbe par classe (one-vs-rest), AUC dans la légende."""
    fig = go.Figure()
    for i, curve in enumerate(roc["curves"]):
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{curve['class']} (AUC {curve['auc']:.2f})",
                line=dict(color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Hasard",
            line=dict(color="#9e9e9e", dash="dash"),
        )
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.05])
    return _to_dict(fig, "Courbe ROC", "Taux de faux positifs", "Taux de vrais positifs", height=420)


def _actual_vs_predicted_chart(predictions: dict) -> dict:
    y_true, y_pred = predictions["y_true"], predictions["y_pred"]
    low, high = min(y_true + y_pred), max(y_true + y_pred)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Prédictions",
            marker=dict(color=PALETTE[0], opacity=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Prédiction parfaite",
            line=dict(color=PALETTE[1], dash="dash"),
        )
    )
    return _to_dict(fig, "Valeurs réelles vs prédites", "Valeur réelle", "Valeur prédite", height=450)


def _residuals_chart(predictions: dict) -> dict:
    """Résidus (réel - prédit) en fonction de la valeur prédite : révèle les biais systématiques."""
    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred"], dtype=float)
    residus = y_true - y_pred

    fig = go.Figure(
        go.Scatter(
            x=y_pred.tolist(),
            y=residus.tolist(),
            mode="markers",
            marker=dict(color=PALETTE[3], opacity=0.6),
            name="Résidus",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9e9e9e")
    return _to_dict(fig, "Résidus", "Valeur prédite", "Erreur (réel - prédit)", height=420)


def _classification_comparison_chart(models: list[dict]) -> dict:
    names = [m["name"] for m in models]
    fig = go.Figure()
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1-score"),
        ("balanced_accuracy", "Accuracy équilibrée"),
        ("mcc", "MCC"),
    ]:
        fig.add_trace(
            go.Bar(name=label, x=names, y=_values(models, lambda m, k=metric: m["metrics"][k]))
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[-1, 1.05])
    return _to_dict(fig, "Comparaison des modèles", "Modèle", "Score")


def _regression_comparison_charts(models: list[dict]) -> dict:
    names = [m["name"] for m in models]

    r2_fig = go.Figure(
        go.Bar(
            x=names,
            y=_values(models, lambda m: m["metrics"]["r2"]),
            marker_color=PALETTE[0],
        )
    )
    r2 = _to_dict(r2_fig, "Comparaison des modèles : R²", "Modèle", "R²")

    error_fig = go.Figure()
    for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("median_ae", "Erreur médiane")]:
        error_fig.add_trace(
            go.Bar(name=label, x=names, y=_values(models, lambda m, k=metric: m["metrics"][k]))
        )
    error_fig.update_layout(barmode="group")
    errors = _to_dict(error_fig, "Comparaison des modèles : erreurs (plus bas = mieux)", "Modèle", "Erreur")

    # MAPE à part : exprimée en %, elle n'a pas la même échelle que les erreurs absolues
    mape_fig = go.Figure(
        go.Bar(
            x=names,
            y=_values(models, lambda m: m["metrics"]["mape"]),
            marker_color=PALETTE[4],
        )
    )
    mape = _to_dict(mape_fig, "Comparaison des modèles : MAPE (%)", "Modèle", "MAPE (%)")

    return {"r2": r2, "errors": errors, "mape": mape}


def build_supervised_charts(result: dict) -> dict:
    """Graphiques à partir du résultat de run_supervised()."""
    models = result["models"]
    is_classification = result["problem_type"] == "classification"

    if is_classification:
        comparison = {"metrics": _classification_comparison_chart(models)}
    else:
        comparison = _regression_comparison_charts(models)

    per_model = {}
    for model in models:
        charts = {"feature_importance": _feature_importance_chart(model["feature_importance"])}
        if is_classification:
            charts["confusion_matrix"] = _confusion_matrix_chart(model["metrics"]["confusion_matrix"])
            charts["per_class"] = _per_class_chart(model["metrics"]["per_class"])
            if model.get("roc"):
                charts["roc"] = _roc_chart(model["roc"])
        else:
            charts["actual_vs_predicted"] = _actual_vs_predicted_chart(model["predictions"])
            charts["residuals"] = _residuals_chart(model["predictions"])
        per_model[model["key"]] = charts

    return {
        "best_model": result["best_model"],
        "comparison": comparison,
        "models": per_model,
    }


# ---------------------------------------------------------------------------
# Non supervisé
# ---------------------------------------------------------------------------
def _cluster_scatter(projection: dict) -> dict:
    x = np.array(projection["x"])
    y = np.array(projection["y"])
    labels = np.array(projection["labels"])
    var1, var2 = projection["explained_variance"]

    fig = go.Figure()
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        fig.add_trace(
            go.Scatter(
                x=x[mask].tolist(),
                y=y[mask].tolist(),
                mode="markers",
                name=_cluster_name(cluster),
                marker=dict(color=_cluster_color(cluster), opacity=0.7, size=7),
            )
        )
    return _to_dict(
        fig,
        "Clusters (projection PCA)",
        f"Axe 1 ({var1 * 100:.0f} % de la variance)",
        f"Axe 2 ({var2 * 100:.0f} % de la variance)",
        height=480,
    )


def _cluster_sizes_chart(profiles: list[dict]) -> dict:
    fig = go.Figure(
        go.Bar(
            x=[_cluster_name(p["cluster"]) for p in profiles],
            y=[p["size"] for p in profiles],
            text=[f"{p['percentage']} %" for p in profiles],
            textposition="outside",
            marker_color=[_cluster_color(p["cluster"]) for p in profiles],
        )
    )
    return _to_dict(fig, "Taille des clusters", "Cluster", "Nombre de lignes", height=350)


def _k_selection_charts(k_selection: list[dict]) -> dict | None:
    if not k_selection:
        return None
    ks = [item["k"] for item in k_selection]

    elbow = go.Figure(
        go.Scatter(
            x=ks,
            y=[item["inertia"] for item in k_selection],
            mode="lines+markers",
            marker_color=PALETTE[0],
        )
    )
    elbow.update_xaxes(dtick=1)

    silhouette = go.Figure(
        go.Scatter(
            x=ks,
            y=[item["silhouette"] for item in k_selection],
            mode="lines+markers",
            marker_color=PALETTE[2],
        )
    )
    silhouette.update_xaxes(dtick=1)

    return {
        "elbow": _to_dict(elbow, "Méthode du coude (K-Means)", "Nombre de clusters (k)", "Inertie", height=350),
        "silhouette": _to_dict(
            silhouette, "Silhouette selon k (K-Means)", "Nombre de clusters (k)", "Silhouette", height=350
        ),
    }


def _clustering_comparison_chart(models: list[dict]) -> dict:
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in models],
            y=[m["metrics"]["silhouette"] for m in models],
            marker_color=PALETTE[0],
        )
    )
    return _to_dict(fig, "Comparaison des algorithmes : silhouette", "Algorithme", "Silhouette", height=350)


def build_unsupervised_charts(result: dict) -> dict:
    """Graphiques à partir du résultat de run_unsupervised()."""
    models = result["models"]

    per_model = {
        model["key"]: {
            "scatter": _cluster_scatter(model["projection"]),
            "cluster_sizes": _cluster_sizes_chart(model["profiles"]),
        }
        for model in models
    }

    return {
        "best_model": result["best_model"],
        "comparison": {"silhouette": _clustering_comparison_chart(models)},
        "k_selection": _k_selection_charts(result["k_selection"]),
        "models": per_model,
    }