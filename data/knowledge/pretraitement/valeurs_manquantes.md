# Traitement des valeurs manquantes

## En bref
Les valeurs manquantes sont remplacées par une valeur estimée à partir des données d'entraînement.
Pour remplir les trous d’une colonne numérique, l’application utilise la médiane de cette colonne.

## Explication
- Colonne numérique : remplacement par la médiane. La médiane est peu sensible aux valeurs extrêmes.
- Colonne catégorielle : remplacement par la valeur la plus fréquente.
- Une colonne dont plus de 60 % des valeurs manquent est retirée complètement, car son remplacement serait trop incertain.
- Les lignes sans valeur cible sont retirées en supervisé, car elles ne peuvent pas servir à l'entraînement.

## Dans DataAnalyzer
Le nombre de valeurs manquantes par colonne est affiché dans le rapport PDF et dans l'aperçu du dataset.

## Limites
Remplacer une valeur manquante par une estimation réduit l'information disponible. Une valeur manquante peut aussi avoir un sens en elle-même.