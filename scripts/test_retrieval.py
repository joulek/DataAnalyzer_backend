"""Vérifie que les bons documents remontent pour quelques questions (français et anglais).
À lancer depuis backend/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.retriever import retrieve  # noqa: E402

# (question, document attendu en premier)
CASES = [
    # --- Français ---
    ("Comment est calculé le score ?", "application/score.md"),
    ("Pourquoi le meilleur modèle est-il choisi sur le F1 ?", "application/choix_meilleur_modele.md"),
    ("Qu'est-ce que la fuite de données ?", "pretraitement/fuite_de_donnees.md"),
    ("Quelle est la différence entre MAE et RMSE ?", "metriques/r2_mae_rmse.md"),
    ("Pourquoi le SVM n'est-il pas entraîné sur tous les datasets ?", "algorithmes/svm.md"),
    ("Que signifie une silhouette négative ?", "metriques/silhouette_davies_bouldin.md"),
    ("Comment les valeurs aberrantes sont-elles traitées ?", "pretraitement/valeurs_aberrantes.md"),
    ("Qu'est-ce que le surapprentissage ?", "ml/surapprentissage.md"),
    ("Qu'est-ce que le clustering ?", "algorithmes/kmeans.md"),
    ("Pourquoi équilibrer les classes ?", "ml/classes_desequilibrees.md"),
    ("Comment fonctionne la régression logistique ?", "algorithmes/regression_logistique.md"),
    ("À quoi sert la matrice de confusion ?", "metriques/matrice_confusion.md"),
    # --- Anglais (mêmes documents attendus) ---
    ("How is the score calculated?", "application/score.md"),
    ("Why is the best model chosen on F1?", "application/choix_meilleur_modele.md"),
    ("What is data leakage?", "pretraitement/fuite_de_donnees.md"),
    ("What is the difference between MAE and RMSE?", "metriques/r2_mae_rmse.md"),
    ("Why is the SVM not trained on all datasets?", "algorithmes/svm.md"),
    ("What does a negative silhouette mean?", "metriques/silhouette_davies_bouldin.md"),
    ("How are outliers handled?", "pretraitement/valeurs_aberrantes.md"),
    ("What is overfitting?", "ml/surapprentissage.md"),
    ("What is clustering?", "algorithmes/kmeans.md"),
    ("Why balance the classes?", "ml/classes_desequilibrees.md"),
    ("How does logistic regression work?", "algorithmes/regression_logistique.md"),
    ("What is a confusion matrix used for?", "metriques/matrice_confusion.md"),
]

if __name__ == "__main__":
    results = {"fr": [0, 0], "en": [0, 0]}  # [réussies, total]

    for index, (question, expected) in enumerate(CASES):
        lang = "fr" if index < len(CASES) // 2 else "en"
        passages = retrieve(question, top_k=2)
        top = passages[0]["source"] if passages else None
        status = "OK" if top == expected else "KO"

        results[lang][1] += 1
        results[lang][0] += status == "OK"

        print(f"\n[{status}] ({lang}) {question}")
        print(f"      attendu : {expected}")
        for passage in passages:
            print(f"      [{passage['score']}] {passage['source']} > {passage['section']}")

    total_ok = results["fr"][0] + results["en"][0]
    total = results["fr"][1] + results["en"][1]
    print(f"\nFrançais : {results['fr'][0]}/{results['fr'][1]}")
    print(f"Anglais  : {results['en'][0]}/{results['en'][1]}")
    print(f"Total    : {total_ok}/{total} questions avec le bon document en tête.")