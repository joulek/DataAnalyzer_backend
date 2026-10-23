# Transformation logarithmique

## En bref
Une variable très asymétrique et positive est transformée par logarithme pour réduire l'influence de ses valeurs très élevées.

## Explication
L'asymétrie (skewness) mesure si la distribution penche d'un côté. Une valeur supérieure à 1 indique une queue de distribution très longue vers la droite, comme les revenus ou les prix.

Pour ces variables, l'application applique log(1 + x). La transformation n'est appliquée que si toutes les valeurs sont positives ou nulles.

Le seuil d'asymétrie et les paramètres sont appris sur les données d'entraînement.

## Dans DataAnalyzer
La transformation se fait avant le plafonnement IQR et la standardisation. Elle est particulièrement utile pour la régression linéaire.

## Limites
Après transformation, les coefficients d'un modèle linéaire ne s'interprètent plus de la même manière que sur la valeur d'origine.