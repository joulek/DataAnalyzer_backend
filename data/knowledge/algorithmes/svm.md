# Machine à vecteurs de support (SVM)

## En bref
Un modèle qui cherche la frontière de séparation la plus éloignée possible des points de chaque classe. Avec le noyau RBF, il peut suivre des frontières courbes.

## Explication
Le noyau RBF transforme implicitement les données pour rendre possible une séparation non linéaire. Le paramètre de régularisation C est fixé à 1,0.

En classification, les classes sont équilibrées. En régression, la version utilisée est le SVR.

Le SVM a un coût de calcul qui augmente vite avec le nombre de lignes. L'application ne l'entraîne donc que sur 5 000 lignes d'entraînement au maximum : au-delà, il apparaît dans les modèles non entraînés, avec la raison.

## Dans DataAnalyzer
Disponible en classification et en régression. Il ne fournit pas de probabilités par défaut, donc pas de courbe ROC ni d'AUC.

## Limites
Sans probabilités, on ne peut pas ajuster le seuil de décision. Le réglage de C et du noyau influence fortement le résultat.