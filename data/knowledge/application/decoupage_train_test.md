# Découpage entre entraînement et test

## En bref
Les données sont séparées en 80 % pour l'entraînement et 20 % pour le test. La séparation est reproductible grâce à une graine fixe de 42.

## Explication
- Le jeu de test n'est jamais vu pendant l'entraînement. Il sert à mesurer la performance sur des données nouvelles.
- En classification, le découpage est stratifié quand c'est possible : chaque classe garde à peu près la même proportion dans les deux parties.
- Une classe qui ne contient qu'une seule observation est retirée avant le découpage, car elle ne peut pas être répartie.
- Pour le dataset Iris (150 lignes), on obtient 120 lignes d'entraînement et 30 lignes de test.

## Dans DataAnalyzer
Le découpage est réalisé dans le service d'entraînement supervisé. Le nombre de lignes d'entraînement et de test est renvoyé dans les résultats.

## Limites
Un seul découpage peut être chanceux ou malchanceux, surtout sur un petit jeu de données. La validation croisée donne une estimation plus stable.