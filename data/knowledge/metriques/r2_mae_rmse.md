# R², MAE et RMSE

## En bref
Ces métriques mesurent la qualité d'un modèle de régression, soit par la part de variation expliquée (R²), soit par l'erreur moyenne dans l'unité de la cible (MAE et RMSE).

## Explication
- R² : part de la variation de la cible expliquée par le modèle. 1 est parfait, 0 équivaut à prédire la moyenne, une valeur négative est pire que la moyenne.
- MAE (erreur absolue moyenne) : moyenne des écarts absolus, dans l'unité de la cible.
- RMSE (racine de l'erreur quadratique moyenne) : pénalise davantage les grosses erreurs que la MAE. Elle est toujours supérieure ou égale à la MAE.

Une erreur médiane et un MAPE (erreur en pourcentage) sont aussi calculés pour compléter l'analyse.

## Dans DataAnalyzer
Le meilleur modèle de régression est choisi sur le R². Le score est ce R² ramené entre 0 et 100 %.

## Limites
Un R² élevé ne garantit pas que les prédictions soient utilisables. Une MAE de 5 000 € peut être acceptable pour une maison à 500 000 € et inacceptable pour un produit à 20 €.