# Accuracy, précision, rappel et F1-score

## En bref
Ces quatre métriques mesurent les bonnes décisions d'un modèle de classification, chacune sous un angle différent.
Pourquoi ne pas se fier seulement à l’accuracy ? Parce qu’elle paraît bonne quand une classe domine, même si le modèle ne prédit pas bien les cas rares.

## Explication
- Accuracy : part de prédictions correctes. Trompeuse si une classe domine.
- Precision : parmi les prédictions d'une classe, part de celles qui sont justes.
- Recall (rappel) : parmi les vrais cas d'une classe, part de ceux que le modèle a trouvés.
- F1-score : moyenne harmonique de la precision et du recall. Elle pénalise un modèle qui sacrifie l'un pour l'autre.

Les versions « pondérée » tiennent compte de la taille de chaque classe. La version « macro » donne le même poids à chaque classe.

## Dans DataAnalyzer
Le meilleur modèle en classification est choisi sur le F1 pondéré. Le détail par classe est affiché dans les graphiques de chaque modèle.

## Limites
Un bon F1 global peut cacher une classe rare mal prédite. Regardez le détail par classe.