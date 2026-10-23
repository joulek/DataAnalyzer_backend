# Les classes déséquilibrées

## En bref
Un jeu de données est déséquilibré quand une classe est beaucoup plus fréquente que les autres. Le modèle a alors tendance à privilégier la classe majoritaire.
Les classes rares sont des classes peu représentées dans les données.

## Explication
Exemple : si 95 % des clients ne résilient pas, un modèle qui répond toujours « non » a 95 % d'accuracy, sans rien apprendre.

Pour limiter ce problème, l'application :
- équilibre les classes dans la régression logistique, le Random Forest, l'arbre de décision, le Gradient Boosting et le SVM (`class_weight="balanced"`) ;
- utilise le F1 pondéré pour choisir le meilleur modèle, plutôt que l'accuracy ;
- affiche un avertissement quand la plus petite classe représente moins de 10 % de la plus grande.

## Dans DataAnalyzer
L'avertissement apparaît dans l'étape de détection du type de problème, et le détail par classe est affiché pour chaque modèle.

## Limites
Équilibrer les poids ne crée pas de nouvelles informations. Si la classe rare est peu représentée, le modèle a peu d'exemples pour apprendre.