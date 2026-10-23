# Le rapport PDF et le notebook exportés

## En bref
L'application peut exporter les résultats en deux formats : un rapport PDF pour la lecture, et un notebook Jupyter pour reproduire l'analyse.

## Explication
Le rapport PDF contient :
- un résumé : type de problème, cible, meilleur modèle et son score ;
- les données : lignes et colonnes avant et après nettoyage, valeurs manquantes ;
- le prétraitement : les étapes appliquées et les colonnes retirées ;
- la comparaison des modèles et le détail de chacun.

Le notebook contient les mêmes étapes sous forme de code Python exécutable : chargement, nettoyage, prétraitement, entraînement et évaluation. Il ne dépend pas du backend et s'exécute avec pandas, scikit-learn, matplotlib et openpyxl (pour un fichier Excel).

## Dans DataAnalyzer
Les deux exports sont disponibles sur la page Résultats, à côté du bouton de nouvelle analyse. Le score du notebook doit être identique à celui de l'application.

## Limites
Les graphiques ne sont pas tous reproduits dans le notebook. Le PDF ne contient pas les graphiques d'exploration interactifs.