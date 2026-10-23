# Le surapprentissage

## En bref
Le surapprentissage se produit quand un modèle mémorise les données d'entraînement, y compris leur bruit, et généralise mal sur de nouvelles données.
Un modèle qui apprend par cœur ses données fait du surapprentissage (overfitting).

## Explication
Symptômes typiques : un score très élevé sur l'entraînement et nettement plus bas sur le test. Un modèle trop complexe, comme un arbre sans limite de profondeur, est plus exposé.

Moyens de le réduire : simplifier le modèle, avoir plus de données, limiter la profondeur des arbres (8 dans l'application), ou utiliser la validation croisée pour mieux estimer la performance réelle.

## Dans DataAnalyzer
L'application ne calcule pas encore le score d'entraînement. Un écart important entre les scores de deux modèles sur le même découpage peut toutefois être un indice.

## Limites
Un score de test élevé ne prouve pas l'absence de surapprentissage, surtout sur un petit jeu de test.