# Le score de 0 à 100 %

## En bref
Le score résume la performance d'un modèle sur une seule échelle, de 0 à 100 %.
Il est calculé à partir de la métrique principale de chaque type de problème.
Le score du modèle est un chiffre de 0 à 100 % qui résume sa performance.

## Explication
- Classification : F1-score pondéré × 100.
- Régression : R² × 100. Un R² négatif donne un score de 0 %, car le modèle fait pire que la moyenne.
- Clustering : silhouette × 100. Une silhouette négative donne 0 %.

Le score ne sert qu'à comparer les modèles d'une même analyse. Il ne dit rien de la qualité absolue
de vos données.

## Dans DataAnalyzer
Le score est enregistré pour chaque modèle. Le meilleur modèle est celui dont la métrique principale
est la plus élevée, et il est affiché en tête de la page Résultats.

## Limites
Un score de 95 % sur un petit jeu de test peut varier beaucoup avec un autre découpage.
La validation croisée donne une estimation plus stable.