# La régression logistique

## En bref
Un modèle linéaire qui estime la probabilité d'appartenir à chaque classe. Simple, rapide et facile à interpréter.

## Explication
Le modèle calcule une combinaison linéaire des variables, puis la transforme en probabilité entre 0 et 1. Dans l'application, la version utilisée équilibre automatiquement les classes (`class_weight="balanced"`) et accepte jusqu'à 1000 itérations d'optimisation.

Les variables numériques sont standardisées avant l'entraînement, ce qui est nécessaire pour que le modèle converge correctement.

## Dans DataAnalyzer
Disponible uniquement en classification. Sa forme simple en fait souvent un bon point de comparaison.

## Limites
Elle suppose une frontière de décision à peu près linéaire. Elle peut être dépassée par des modèles d'arbres sur des relations complexes.