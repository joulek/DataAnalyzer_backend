# Mélange de gaussiennes (GMM)

## En bref
Un algorithme qui suppose que les données viennent de plusieurs distributions gaussiennes. Chaque point reçoit une probabilité d'appartenance à chaque groupe.

## Explication
Contrairement à K-Means, chaque groupe peut avoir une forme elliptique et une taille propre. Le nombre de composantes est le même que k. Le modèle utilise une matrice de covariance complète, ce qui lui permet de capturer des formes obliques.

Dans l'application, l'affectation finale suit la probabilité la plus élevée.

## Dans DataAnalyzer
Disponible en clustering, avec le même k que K-Means pour une comparaison directe.

## Limites
Il peut produire des groupes très déséquilibrés. Surveillez la taille des clusters dans le détail du modèle.