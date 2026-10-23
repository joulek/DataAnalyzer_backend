"""Résumé compact d'une analyse, transmis au LLM comme contexte (jamais le dataset)."""


def summarize_analysis(analysis: dict) -> str:
    detection = analysis.get("detection", {})
    results = analysis.get("results", {})
    models = results.get("models", [])
    best_key = results.get("best_model")

    lines = [
        f"Type de problème : {results.get('problem_type')}",
        f"Apprentissage : {detection.get('learning_type')}",
        f"Colonne cible : {detection.get('target') or 'aucune'}",
        f"Meilleur modèle : {best_key} (score {results.get('best_score')} %)",
        "Modèles comparés :",
    ]
    for m in models:
        metrics = m.get("metrics", {})
        keys = ["f1_weighted", "accuracy", "mcc", "roc_auc", "r2", "rmse", "silhouette"]
        shown = {k: metrics[k] for k in keys if k in metrics and metrics[k] is not None}
        lines.append(f"- {m.get('name')} : score {m.get('score')} %, métriques {shown}")

    dropped = analysis.get("dropped_columns", [])
    if dropped:
        lines.append("Colonnes retirées : " + ", ".join(f"{d['column']} ({d['reason']})" for d in dropped))

    failed = results.get("failed_models", [])
    if failed:
        lines.append("Modèles non entraînés : " + ", ".join(f"{f['model']} ({f['error']})" for f in failed))

    return "\n".join(lines)