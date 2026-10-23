"""Génération du rapport PDF : données, prétraitement et résultats par modèle."""

from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services.preprocessing import (
    MAX_CATEGORIES,
    MIN_ROWS,
    MISSING_COLUMN_THRESHOLD,
    clean_dataframe,
)
from app.services.scoring import SCORE_METRIC_LABEL
from app.services.supervised import RANDOM_STATE, TEST_SIZE
from app.services.unsupervised import (
    MAX_AGGLOMERATIVE_ROWS,
    MAX_K,
    MIN_K,
    SILHOUETTE_SAMPLE_SIZE,
)

# Palette de l'application
INK = colors.HexColor("#1f1d2b")
ACCENT = colors.HexColor("#ff5a36")
SOFT = colors.HexColor("#fdeee2")
LINE = colors.HexColor("#ecdfd1")
MUTED = colors.HexColor("#6e6a7c")

USABLE_WIDTH = A4[0] - 4 * cm

TYPE_LABELS = {
    "classification": "Classification",
    "regression": "Régression",
    "clustering": "Clustering",
}

METRIC_COLUMNS = {
    "classification": [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1 pondéré"),
    ],
    "regression": [("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE")],
    "clustering": [
        ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("calinski_harabasz", "Calinski-Harabasz"),
    ],
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Accuracy équilibrée",
    "precision_weighted": "Precision (pondérée)",
    "recall_weighted": "Recall (pondéré)",
    "f1_weighted": "F1-score (pondéré)",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1-score (macro)",
    "r2": "R²",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
}

_base = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_base["Title"], textColor=INK, alignment=0, fontSize=22, leading=26, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=_base["Heading2"], textColor=INK, fontSize=14, spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=_base["Heading3"], textColor=ACCENT, fontSize=11, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=_base["BodyText"], fontSize=9.5, leading=13, textColor=INK)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELL_BOLD = ParagraphStyle("CellBold", parent=CELL, fontName="Helvetica-Bold")
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)


# ---------------------------------------------------------------------------
# Outils de mise en forme
# ---------------------------------------------------------------------------
def _p(text, style=CELL):
    return Paragraph(escape(str(text)), style)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Oui" if value else "Non"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
    return str(value)


def _score(value) -> str:
    return "—" if value is None else f"{float(value):.1f} %"


def _widths(n: int, first: float = 3.6 * cm) -> list[float]:
    rest = (USABLE_WIDTH - first) / (n - 1)
    return [first] + [rest] * (n - 1)


def _table(header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> Table:
    if widths is None:
        widths = [USABLE_WIDTH / len(header)] * len(header)
    data = [[_p(h, CELL_HEAD) for h in header]]
    data += [[_p(cell) for cell in row] for row in rows]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _kv(rows: list[tuple[str, str]]) -> Table:
    """Tableau à deux colonnes : libellé en gras, valeur à droite."""
    data = [[_p(k, CELL_BOLD), _p(v)] for k, v in rows]
    table = Table(data, colWidths=[5.5 * cm, USABLE_WIDTH - 5.5 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), SOFT),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _bullets(story: list, items: list[str]) -> None:
    for item in items:
        story.append(Paragraph(item, BULLET, bulletText="•"))


# ---------------------------------------------------------------------------
# Résumé du dataset (recalculé à partir du fichier, pour les chiffres bruts)
# ---------------------------------------------------------------------------
def summarize_dataset(df: pd.DataFrame, target: str | None) -> dict:
    cleaned = clean_dataframe(df)

    rows_without_target = 0
    if target and target in cleaned.columns:
        rows_without_target = int(cleaned[target].isna().sum())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    return {
        "raw_rows": len(df),
        "raw_columns": df.shape[1],
        "clean_rows": len(cleaned),
        "clean_columns": cleaned.shape[1],
        "duplicates": len(df) - len(cleaned),
        "empty_columns": df.shape[1] - cleaned.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": [
            (str(col), int(count), round(count / len(df) * 100, 1))
            for col, count in missing.items()
        ],
        "rows_without_target": rows_without_target,
    }


# ---------------------------------------------------------------------------
# Sections du rapport
# ---------------------------------------------------------------------------
def _summary_section(story: list, analysis: dict) -> None:
    detection = analysis["detection"]
    results = analysis["results"]
    models = results.get("models", [])
    best = next((m for m in models if m["key"] == results.get("best_model")), None)

    story.append(Paragraph("1. Résumé", H2))
    story.append(
        _kv(
            [
                ("Type de problème", TYPE_LABELS.get(results["problem_type"], results["problem_type"])),
                (
                    "Apprentissage",
                    "Supervisé" if detection["learning_type"] == "supervised" else "Non supervisé",
                ),
                ("Colonne cible", detection.get("target") or "Aucune (clustering)"),
                ("Justification", detection.get("reason", "")),
                ("Meilleur modèle", best["name"] if best else "—"),
                ("Score du meilleur modèle", _score(results.get("best_score"))),
                ("Modèles comparés", str(len(models))),
            ]
        )
    )
    warnings = detection.get("warnings", [])
    if warnings:
        story.append(Spacer(1, 6))
        _bullets(story, [escape(w) for w in warnings])


def _data_section(story: list, analysis: dict, summary: dict) -> None:
    is_supervised = analysis["detection"]["learning_type"] == "supervised"

    story.append(Paragraph("2. Données", H2))
    rows = [
        ("Lignes (fichier brut)", str(summary["raw_rows"])),
        ("Colonnes (fichier brut)", str(summary["raw_columns"])),
        ("Doublons supprimés", str(summary["duplicates"])),
        ("Colonnes vides supprimées", str(summary["empty_columns"])),
        ("Lignes après nettoyage", str(summary["clean_rows"])),
        ("Valeurs manquantes (total)", str(summary["missing_total"])),
    ]
    if is_supervised:
        rows.append(("Lignes sans cible retirées", str(summary["rows_without_target"])))
    story.append(_kv(rows))

    if summary["missing_by_column"]:
        story.append(Paragraph("Valeurs manquantes par colonne", H3))
        story.append(
            _table(
                ["Colonne", "Manquantes", "Part (%)"],
                [[col, str(n), f"{pct} %"] for col, n, pct in summary["missing_by_column"]],
                widths=[9 * cm, 4 * cm, 4 * cm],
            )
        )


def _preprocessing_section(story: list, analysis: dict, summary: dict) -> None:
    detection = analysis["detection"]
    results = analysis["results"]
    is_supervised = detection["learning_type"] == "supervised"
    problem = results["problem_type"]

    story.append(Paragraph("3. Prétraitement", H2))

    steps = [
        f"<b>Nettoyage :</b> suppression des lignes dupliquées ({summary['duplicates']}) "
        f"et des colonnes entièrement vides ({summary['empty_columns']}).",
        f"<b>Colonnes retirées :</b> colonnes de type date, colonnes avec plus de "
        f"{MISSING_COLUMN_THRESHOLD:.0%} de valeurs manquantes, colonnes constantes, "
        f"et textes libres de plus de {MAX_CATEGORIES} valeurs différentes.",
        "<b>Valeurs manquantes restantes :</b> médiane pour les colonnes numériques, "
        "valeur la plus fréquente pour les colonnes catégorielles.",
        "<b>Transformation :</b> standardisation des colonnes numériques (moyenne 0, "
        "écart-type 1) et encodage One-Hot des colonnes catégorielles.",
        f"<b>Taille minimale :</b> {MIN_ROWS} lignes exploitables.",
    ]

    if is_supervised:
        steps.append(
            f"<b>Découpage :</b> {TEST_SIZE:.0%} des données pour le test et "
            f"{1 - TEST_SIZE:.0%} pour l'entraînement (graine aléatoire {RANDOM_STATE}), "
            f"stratifié quand c'est possible. Entraînement : {results.get('n_train')} lignes, "
            f"test : {results.get('n_test')} lignes."
        )
        steps.append(
            "<b>Anti-fuite :</b> le préprocesseur est entraîné uniquement sur les données "
            "d'entraînement, jamais sur les données de test."
        )
        if problem == "classification":
            steps.append(
                "<b>Classes :</b> les classes ne contenant qu'une seule observation sont retirées, "
                "puis la cible est encodée en entiers."
            )
            steps.append(
                "<b>Équilibrage :</b> pour la régression logistique, le Random Forest et l'arbre, "
                "les classes rares reçoivent un poids plus élevé (class_weight = balanced)."
            )
    else:
        steps.append(
            "<b>Données complètes :</b> le préprocesseur est appliqué à l'ensemble du jeu, "
            "sans découpage."
        )
        choice = (
            "choix automatique"
            if results.get("auto_k")
            else f"valeur imposée : {results.get('k')}"
        )
        steps.append(
            f"<b>Nombre de clusters :</b> entre {MIN_K} et {MAX_K}, selon la silhouette de "
            f"K-Means ({choice})."
        )
        steps.append(
            f"<b>Silhouette :</b> estimée sur {SILHOUETTE_SAMPLE_SIZE} points au maximum "
            "pour les grands jeux de données."
        )
        steps.append(
            f"<b>Clustering hiérarchique :</b> limité à {MAX_AGGLOMERATIVE_ROWS} lignes."
        )
        steps.append(
            "<b>DBSCAN :</b> le paramètre eps est estimé à partir de la distance au k-ième voisin."
        )

    _bullets(story, steps)

    dropped = analysis.get("dropped_columns", [])
    if dropped:
        story.append(Paragraph("Colonnes retirées", H3))
        story.append(
            _table(
                ["Colonne", "Raison"],
                [[d["column"], d["reason"]] for d in dropped],
                widths=[6 * cm, USABLE_WIDTH - 6 * cm],
            )
        )


def _model_row(model: dict, problem: str, best_key: str | None) -> list[str]:
    name = model["name"] + (" (meilleur)" if model["key"] == best_key else "")
    row = [name, _score(model.get("score"))]
    row += [_fmt(model.get("metrics", {}).get(key)) for key, _ in METRIC_COLUMNS[problem]]
    if problem == "clustering":
        row += [_fmt(model.get("n_clusters")), _fmt(model.get("n_noise"))]
    return row


def _comparison_section(story: list, analysis: dict) -> None:
    results = analysis["results"]
    problem = results["problem_type"]
    best_key = results.get("best_model")
    models = results.get("models", [])

    story.append(Paragraph("4. Comparaison des modèles", H2))
    story.append(
        Paragraph(escape(f"Score basé sur : {SCORE_METRIC_LABEL.get(problem, '')}."), BODY)
    )
    story.append(Spacer(1, 6))

    header = ["Modèle", "Score"] + [label for _, label in METRIC_COLUMNS[problem]]
    if problem == "clustering":
        header += ["Clusters", "Bruit"]

    rows = [_model_row(m, problem, best_key) for m in models]
    story.append(_table(header, rows, widths=_widths(len(header))))

    failed = results.get("failed_models", [])
    if failed:
        story.append(Paragraph("Modèles non entraînés", H3))
        story.append(
            _table(
                ["Modèle", "Erreur"],
                [[f["model"], f["error"]] for f in failed],
                widths=[5 * cm, USABLE_WIDTH - 5 * cm],
            )
        )


def _model_details(story: list, model: dict, problem: str, is_best: bool) -> None:
    title = model["name"] + ("  (meilleur modèle)" if is_best else "")
    story.append(Paragraph(escape(title), H3))

    metrics = model.get("metrics", {})
    rows = [("Score", _score(model.get("score")))]

    if problem == "clustering":
        rows.append(("Clusters", _fmt(model.get("n_clusters"))))
        rows.append(("Points de bruit", _fmt(model.get("n_noise"))))
    if "train_time_s" in model:
        rows.append(("Temps d'entraînement", f"{model['train_time_s']} s"))

    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            rows.append((METRIC_LABELS.get(key, key.replace("_", " ")), _fmt(value)))

    for key, value in model.get("params", {}).items():
        rows.append((f"Paramètre {key}", _fmt(value)))

    story.append(_kv(rows))

    # Détail par classe et matrice de confusion (classification)
    per_class = metrics.get("per_class")
    if per_class:
        story.append(Paragraph("Métriques par classe", H3))
        story.append(
            _table(
                ["Classe", "Precision", "Recall", "F1", "Support"],
                [
                    [c["class"], _fmt(c["precision"]), _fmt(c["recall"]), _fmt(c["f1"]), str(c["support"])]
                    for c in per_class
                ],
                widths=_widths(5, first=5 * cm),
            )
        )

    confusion = metrics.get("confusion_matrix")
    if confusion and len(confusion["labels"]) <= 10:
        labels = confusion["labels"]
        story.append(Paragraph("Matrice de confusion (lignes : réel, colonnes : prédit)", H3))
        story.append(
            _table(
                ["Réel / Prédit"] + labels,
                [[labels[i]] + [str(v) for v in row] for i, row in enumerate(confusion["matrix"])],
                widths=_widths(len(labels) + 1, first=3.6 * cm),
            )
        )

    # Importance des variables (modèles à base d'arbres)
    importances = model.get("feature_importance")
    if importances:
        story.append(Paragraph("Importance des variables", H3))
        story.append(
            _table(
                ["Variable", "Importance"],
                [[f["feature"], _fmt(f["importance"])] for f in importances],
                widths=[10 * cm, USABLE_WIDTH - 10 * cm],
            )
        )

    # Profil des clusters (clustering)
    profiles = model.get("profiles")
    if profiles:
        story.append(Paragraph("Profil des clusters", H3))
        profile_rows = []
        for p in profiles:
            name = "Bruit" if p["is_noise"] else f"Cluster {p['cluster']}"
            means = "; ".join(f"{k} : {_fmt(v)}" for k, v in p["means"].items()) or "—"
            profile_rows.append([name, str(p["size"]), f"{p['percentage']} %", means])
        story.append(
            _table(
                ["Groupe", "Taille", "Part", "Moyennes des variables"],
                profile_rows,
                widths=[3 * cm, 2.2 * cm, 2.2 * cm, USABLE_WIDTH - 7.4 * cm],
            )
        )


def _details_section(story: list, analysis: dict) -> None:
    results = analysis["results"]
    problem = results["problem_type"]
    best_key = results.get("best_model")

    story.append(Paragraph("5. Détail par modèle", H2))
    for model in results.get("models", []):
        _model_details(story, model, problem, model["key"] == best_key)


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "DataAnalyzer - Rapport d'analyse")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def build_report_pdf(analysis: dict, summary: dict, filename: str) -> bytes:
    """Construit le PDF complet et renvoie son contenu en octets."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Rapport d'analyse",
        author="DataAnalyzer",
    )

    story = []
    story.append(Paragraph("Rapport d'analyse", H1))
    story.append(
        Paragraph(
            escape(f"Dataset : {filename}  |  Généré le {datetime.now():%d/%m/%Y à %H:%M}"),
            BODY,
        )
    )
    story.append(Spacer(1, 4))

    _summary_section(story, analysis)
    _data_section(story, analysis, summary)
    _preprocessing_section(story, analysis, summary)
    _comparison_section(story, analysis)
    _details_section(story, analysis)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()