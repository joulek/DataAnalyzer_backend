# Regroupement des modalités rares

## En bref
Une catégorie qui apparaît très rarement est regroupée dans une catégorie « Autre ».

## Explication
Pour chaque colonne catégorielle :
- une modalité est conservée si elle apparaît dans au moins 1 % des lignes d'entraînement ;
- au maximum 30 modalités sont conservées ;
- toutes les autres sont regroupées dans « Autre ».

Cela évite de créer une colonne par modalité rare, qui n'apporterait presque rien au modèle et augmenterait le risque de surapprentissage.

## Dans DataAnalyzer
Une colonne texte n'est plus retirée seulement parce qu'elle a beaucoup de valeurs : elle est retirée uniquement si elle ressemble à du texte libre ou à un identifiant.

## Limites
Une catégorie rare mais importante pour la cible peut être noyée dans « Autre ».