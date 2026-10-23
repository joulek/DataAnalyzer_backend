# La validation croisée

## En bref
La validation croisée répète l'évaluation sur plusieurs découpages des données, pour obtenir une estimation plus stable de la performance.

## Explication
Avec 5 plis, les données sont découpées en 5 parties. Chaque partie sert de test une fois, pendant que les 4 autres servent à l'entraînement. Le score final est la moyenne des 5 résultats, accompagnée de leur écart-type.

Un écart-type élevé signale un score instable, donc peu fiable.

## Dans DataAnalyzer
L'application utilise actuellement un seul découpage 80/20. La validation croisée est une amélioration prévue, qui donnerait une moyenne et un écart-type par modèle.

## Limites
Elle multiplie le temps de calcul par le nombre de plis.