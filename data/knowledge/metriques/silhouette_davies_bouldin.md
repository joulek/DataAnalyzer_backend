# Silhouette et indice de Davies-Bouldin

## En bref
La silhouette et l'indice de Davies-Bouldin évaluent la qualité d'un clustering sans étiquettes de référence. Une bonne silhouette (proche de 1) signifie des groupes compacts et bien séparés. Une silhouette négative signifie que des points sont probablement rangés dans le mauvais groupe.

## Explication
- Silhouette : comprise entre -1 et 1. Pour chaque point, elle compare la distance moyenne aux autres points de son groupe et la distance moyenne au groupe voisin le plus proche.
  - Proche de 1 : le point est bien à sa place, loin des autres groupes.
  - Proche de 0 : le point est à la frontière entre deux groupes.
  - Négative : le point est plus proche d'un autre groupe que du sien, donc probablement mal classé. Une silhouette moyenne négative signifie que les groupes se chevauchent ou sont mal définis.
- Davies-Bouldin : plus elle est basse, mieux c'est. Elle compare la dispersion de chaque groupe à la distance qui le sépare des autres groupes. Des groupes serrés et éloignés donnent une valeur faible.

## Dans DataAnalyzer
La silhouette sert à choisir le meilleur algorithme de clustering et à déterminer le nombre de clusters avec K-Means. Pour les grands jeux de données, elle est estimée sur un échantillon de 5 000 points au maximum. Une silhouette négative donne un score de 0 % dans l'application.

## Limites
Ces indicateurs favorisent les groupes convexes et bien séparés. Un bon score ne garantit pas que les groupes aient un sens pour votre problème, et une silhouette faible peut être normale pour des données sans structure nette.