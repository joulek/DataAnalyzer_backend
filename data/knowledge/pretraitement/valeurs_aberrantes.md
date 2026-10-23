# Traitement des valeurs aberrantes

## En bref
Les valeurs extrêmes sont ramenées aux bornes IQR, sans être supprimées.

## Explication
La méthode IQR (écart interquartile) calcule :
- Q1 (25e percentile) et Q3 (75e percentile) ;
- IQR = Q3 - Q1 ;
- les bornes : Q1 - 1,5 × IQR et Q3 + 1,5 × IQR.

Une valeur au-delà de ces bornes est ramenée à la borne. La ligne est conservée, donc aucune information n'est perdue. Les bornes sont calculées sur les données d'entraînement uniquement.

Une colonne sans dispersion (IQR égal à zéro) n'est pas plafonnée.

## Dans DataAnalyzer
Le plafonnement fait partie du prétraitement des variables numériques, avant la standardisation.

## Limites
Une valeur extrême peut être une information réelle et importante, pas une erreur. Le plafonnement la réduit volontairement.