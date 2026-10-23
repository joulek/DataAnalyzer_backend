# Pourquoi certaines colonnes sont retirées

## En bref
L'application retire les colonnes qui n'apportent aucune information utile ou qui risquent de fausser le modèle.

## Explication
Une colonne est retirée dans les cas suivants :
- plus de 60 % de valeurs manquantes ;
- une seule valeur distincte (colonne constante) ;
- une valeur qui représente au moins 99 % des lignes (colonne presque constante) ;
- un identifiant : colonne numérique entière dont presque toutes les valeurs sont uniques (au moins 95 %), ou dont le nom se termine par « id » avec des valeurs uniques ;
- un texte libre : colonne texte avec plus de 30 valeurs différentes, dont au moins 50 % sont uniques.

Chaque colonne retirée est listée avec sa raison dans le rapport PDF.

## Dans DataAnalyzer
La liste `dropped_columns` de la réponse d'analyse contient les colonnes retirées et leur raison.

## Limites
Un identifiant mal détecté peut parfois être conservé, ou une variable utile retirée à tort. Vérifiez toujours la liste dans le rapport.