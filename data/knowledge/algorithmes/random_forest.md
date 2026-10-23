# Random Forest

## En bref
Un ensemble de 100 arbres de décision entraînés sur des sous-échantillons différents, dont les prédictions sont combinées. Robuste et souvent précis.

## Explication
Chaque arbre voit une partie aléatoire des données et des variables. Combiner beaucoup d'arbres réduit le surapprentissage qu'aurait un arbre isolé.

Le modèle fournit une importance des variables, calculée à partir de la contribution de chacune à la réduction des erreurs.

## Dans DataAnalyzer
Disponible en classification et en régression, avec 100 arbres. Les 10 variables les plus importantes sont affichées dans le détail du modèle.

## Limites
Il est moins interprétable qu'un arbre unique. Sur de très petits jeux de données, il peut être moins stable.