# K plus proches voisins (KNN)

## En bref
Un modèle qui prédit à partir des cas les plus similaires : il regarde les 5 voisins les plus proches d'un point.

## Explication
- Classification : la classe est choisie par vote des 5 voisins, en pondérant par la distance.
- Régression : la valeur est la moyenne des 5 voisins, pondérée par la distance.

Le KNN n'apprend rien au sens strict : il garde les données et compare chaque nouveau point. C'est pourquoi les variables doivent être à la même échelle, ce que fait le prétraitement.

## Dans DataAnalyzer
Disponible en classification et en régression.

## Limites
Il devient lent sur de grands jeux de données, car il compare chaque point à tous les autres. Il est aussi sensible aux variables peu utiles.