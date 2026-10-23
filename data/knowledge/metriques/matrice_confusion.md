# La matrice de confusion

## En bref
La matrice de confusion montre, pour chaque classe réelle, combien de fois le modèle a prédit chaque classe.
Un vrai positif est un cas positif correctement prédit. Un faux positif est un cas négatif prédit à tort comme positif. Un faux négatif est un cas positif manqué.

## Explication
- Les lignes représentent la classe réelle.
- Les colonnes représentent la classe prédite.
- La diagonale contient les bonnes prédictions. Toutes les autres cases sont des erreurs.

Elle permet de voir quelles classes sont confondues entre elles, ce qu'une seule métrique ne dit pas.

## Dans DataAnalyzer
La matrice est affichée pour chaque modèle de classification, et reproduite dans le rapport PDF pour les problèmes à 10 classes ou moins.

## Limites
Sa lecture devient difficile au-delà de quelques classes.