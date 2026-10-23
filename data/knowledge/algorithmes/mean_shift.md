# Mean Shift

## En bref
Un algorithme qui déplace chaque point vers le centre de la zone la plus dense autour de lui. Il trouve lui-même le nombre de clusters.

## Explication
La largeur de la zone de recherche (bande passante) est estimée à partir des données. Elle détermine la taille des clusters : une bande étroite donne beaucoup de petits clusters, une bande large en donne peu.

Comme le clustering hiérarchique, il compare les points entre eux et est limité à 5 000 lignes.

## Dans DataAnalyzer
Disponible en clustering. Le nombre de clusters trouvé peut être différent de k.

## Limites
Le résultat dépend fortement de la bande passante estimée. Des groupes très déséquilibrés sont fréquents.