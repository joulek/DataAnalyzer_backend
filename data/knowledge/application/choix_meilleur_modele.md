# Comment le meilleur modèle est choisi

## En bref
Le meilleur modèle est celui qui obtient la valeur la plus élevée sur la métrique principale de son type de problème.

## Explication
- Classification : F1-score pondéré (`f1_weighted`).
- Régression : R² (`r2`).
- Clustering : silhouette.

Les autres métriques sont affichées pour comparaison, mais elles ne changent pas le choix du meilleur modèle.
Le score de 0 à 100 % est calculé à partir de cette même métrique, donc le meilleur modèle est aussi celui qui a le meilleur score.

## Dans DataAnalyzer
Le meilleur modèle est indiqué par un trophée dans la liste des modèles de la page Résultats.

## Limites
Un écart faible entre deux modèles n'est pas forcément significatif : il dépend du découpage des données.