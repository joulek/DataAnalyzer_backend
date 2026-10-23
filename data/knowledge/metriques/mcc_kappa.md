# Coefficient de Matthews (MCC) et kappa de Cohen

## En bref
Ces deux métriques mesurent l'accord entre les prédictions et la réalité, en tenant compte du hasard. Elles restent fiables même quand les classes sont très déséquilibrées.
Le MCC sert à mesurer la qualité d’une classification, même quand les classes sont déséquilibrées.

## Explication
- MCC : valeur entre -1 et 1. 1 est une prédiction parfaite, 0 vaut le hasard, -1 est un désaccord total.
- Kappa de Cohen : valeur entre -1 et 1. Elle mesure l'accord au-delà du hasard. Une valeur proche de 0 signifie que le modèle ne fait pas mieux que le hasard.

Contrairement à l'accuracy, elles ne se laissent pas tromper par une classe majoritaire.

## Dans DataAnalyzer
Le MCC est affiché dans le graphique de comparaison des modèles, et les deux métriques sont listées dans le détail de chaque modèle.

## Limites
Elles résument l'ensemble des classes en un seul chiffre. Pour comprendre les erreurs, consultez la matrice de confusion.