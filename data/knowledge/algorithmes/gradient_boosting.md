# Gradient Boosting

## En bref
Un modèle qui construit des arbres successifs, chacun corrigeant les erreurs du précédent. Très performant sur les données tabulaires.

## Explication
Les arbres sont ajoutés un par un (200 itérations dans l'application, avec un taux d'apprentissage de 0,1). Chaque nouvel arbre s'attache aux cas que les précédents prédisent mal.

L'implémentation utilisée (`HistGradientBoosting`) est rapide, même sur plusieurs milliers de lignes, et gère les valeurs manquantes.

## Dans DataAnalyzer
Disponible en classification et en régression. Il est souvent un des meilleurs modèles sur les jeux de données tabulaires.

## Limites
Il est plus sensible au réglage des paramètres qu'un Random Forest. Ses résultats sont aussi moins faciles à expliquer.