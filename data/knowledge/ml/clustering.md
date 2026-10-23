# Le clustering

## En bref
Le clustering regroupe des observations similaires sans utiliser d'étiquettes de référence. Il découvre des groupes que les données contiennent déjà.

## Explication
Contrairement à la classification, aucune cible n'est fournie : l'algorithme cherche lui-même une structure. Les points d'un même groupe se ressemblent, et les groupes sont éloignés les uns des autres.

Plusieurs algorithmes existent, chacun avec sa façon de définir un groupe : K-Means (centres), clustering hiérarchique (fusions successives), DBSCAN (zones denses), Mean Shift (pics de densité), mélange gaussien (distributions).

## Dans DataAnalyzer
Le clustering est utilisé quand aucune colonne cible n'est choisie. Les algorithmes sont comparés avec la silhouette.

## Limites
Sans étiquettes, il n'existe pas de bonne réponse unique. Un groupe n'a de sens que si l'utilisateur peut l'interpréter.

In English: clustering groups similar observations without labels, to discover structure in the data.