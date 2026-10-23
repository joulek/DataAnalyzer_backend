# K-Means

## En bref
Un algorithme de clustering qui range les points dans k groupes, autour de k centres.

## Explication
K-Means place k centres, affecte chaque point au centre le plus proche, puis déplace les centres vers la moyenne de leur groupe. Le processus se répète jusqu'à stabilisation.

Le nombre k doit être fixé. Dans l'application, il est choisi automatiquement entre 2 et 8, avec la silhouette, sauf si vous l'imposez.

## Dans DataAnalyzer
Le graphique de silhouette selon k et la méthode du coude sont affichés dans les résultats de clustering.

## Limites
K-Means suppose des groupes sphériques de taille comparable. Il est sensible aux valeurs extrêmes, d'où l'intérêt du plafonnement IQR.