# Clustering hiérarchique

## En bref
Un algorithme qui fusionne progressivement les groupes les plus proches, jusqu'à obtenir le nombre de clusters souhaité.

## Explication
Dans l'application, la fusion suit le critère de Ward, qui minimise la variance à l'intérieur des groupes. Le nombre de clusters est le même que celui de K-Means.

L'algorithme consomme beaucoup de mémoire, car il compare les points deux à deux. Il est donc limité à 5 000 lignes.

## Dans DataAnalyzer
Disponible en clustering. S'il dépasse la limite de lignes, il apparaît dans les modèles non entraînés.

## Limites
Une fois une fusion faite, elle ne peut pas être annulée. Les résultats dépendent donc de l'ordre des fusions.