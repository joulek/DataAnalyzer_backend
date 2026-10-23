# L'AUC ROC et la courbe ROC

## En bref
L'AUC ROC mesure la capacité du modèle à classer les cas positifs au-dessus des négatifs, quel que soit le seuil de décision.

## Explication
La courbe ROC trace le taux de vrais positifs en fonction du taux de faux positifs, pour tous les seuils possibles.
- AUC = 1 : séparation parfaite.
- AUC = 0,5 : aussi bon que le hasard.
- AUC < 0,5 : le modèle se trompe systématiquement, en inversant les classes.

En multiclasse, l'application trace une courbe par classe (une contre les autres), et l'AUC est une moyenne pondérée.

## Dans DataAnalyzer
L'AUC et la courbe ROC ne sont calculées que si le modèle fournit des probabilités. Le SVM ne les fournit pas par défaut, donc ces valeurs sont absentes pour lui.

## Limites
L'AUC peut rester élevée alors que le modèle est mauvais pour la classe qui vous intéresse, si cette classe est très rare.