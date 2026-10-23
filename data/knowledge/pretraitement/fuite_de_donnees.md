# Fuite de données

## En bref
Une fuite de données se produit quand le modèle voit, pendant l'entraînement, des informations qui ne seront pas disponibles en réalité. Le score devient alors trop optimiste.

## Explication
Exemples de fuites :
- calculer une médiane ou un écart-type sur l'ensemble du fichier, test compris ;
- standardiser les données avant de séparer l'entraînement et le test.

Pour l'éviter, toutes les statistiques apprises (médianes, bornes IQR, seuil d'asymétrie, modalités fréquentes, moyenne et écart-type) sont calculées dans le pipeline, uniquement sur les données d'entraînement.

Les décisions structurelles, comme retirer un identifiant ou convertir un texte en nombre, ne dépendent d'aucune statistique de la cible. Elles sont donc prises sur le fichier complet.

## Dans DataAnalyzer
Le préprocesseur est placé dans le pipeline de chaque modèle. Il est entraîné à chaque fois sur les données d'entraînement.

## Limites
En non supervisé, il n'y a pas de jeu de test : le préprocesseur est appliqué à l'ensemble des données. Le score de clustering ne peut donc pas être interprété comme une performance sur des données nouvelles.