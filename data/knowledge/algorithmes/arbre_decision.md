# L'arbre de décision

## En bref
Un modèle qui pose une suite de questions sur les variables, jusqu'à une décision. Facile à lire et à expliquer.

## Explication
L'arbre découpe les données en groupes de plus en plus homogènes. Dans l'application, la profondeur est limitée à 8 niveaux, pour éviter de mémoriser les données d'entraînement.

En classification, le modèle équilibre les classes (`class_weight="balanced"`).

## Dans DataAnalyzer
Disponible en classification, en régression et, pour l'importance des variables, dans le détail du modèle.

## Limites
Un arbre seul est instable : un petit changement dans les données peut modifier toute sa structure. C'est pour cela que le Random Forest est souvent préféré.