# DBSCAN

## En bref
Un algorithme qui détecte les zones denses de points et isole les points isolés, appelés « bruit ».

## Explication
DBSCAN n'exige pas de fixer le nombre de clusters. Il a besoin de deux paramètres : une distance eps et un nombre minimal de points min_samples.

Dans l'application, eps est estimé automatiquement à partir de la distance au k-ième voisin (90e percentile), et min_samples dépend du nombre de colonnes, entre 5 et 15. Les points de bruit sont exclus du calcul de la silhouette.

## Dans DataAnalyzer
Disponible en clustering. Le nombre de clusters qu'il trouve peut différer du k choisi pour les autres algorithmes.

## Limites
Il est sensible aux densités différentes entre les groupes. Sur des données à dimensions nombreuses, la notion de distance devient moins discriminante.