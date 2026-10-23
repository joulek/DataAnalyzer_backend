# DataAnalyzer : backend

API FastAPI qui analyse des datasets avec le machine learning, génère un rapport PDF et un notebook Jupyter, et répond aux questions via un assistant RAG.

## Fonctionnalités

- **Upload** de fichiers CSV ou XLSX (20 Mo maximum), avec aperçu.
- **Détection automatique** du type de problème : classification, régression ou clustering.
- **Prétraitement** : conversion des nombres écrits en texte, extraction des dates, suppression des identifiants et colonnes inutilisables, regroupement des modalités rares, transformation logarithmique, plafonnement des valeurs aberrantes (IQR), standardisation, encodage One-Hot.
- **Modèles comparés** :
  - classification et régression : régression logistique ou linéaire, Random Forest, arbre de décision, Gradient Boosting, K plus proches voisins, SVM ;
  - clustering : K-Means, clustering hiérarchique, DBSCAN, Mean Shift, mélange gaussien (GMM).
- **Score de 0 à 100 %** pour chaque modèle, basé sur la métrique principale du type de problème.
- **Rapport PDF** et **notebook Jupyter** qui reproduisent l'analyse.
- **Historique** des analyses enregistré dans PostgreSQL.
- **Assistant RAG** : répond à partir d'une base de connaissances (documents Markdown), ou à partir des connaissances générales du LLM quand aucun document ne correspond. Il accepte le contexte de l'analyse en cours et l'historique de la conversation.

## Structure du projet

```
backend/
├── app/
│   ├── main.py              # point d'entrée FastAPI
│   ├── config.py            # configuration générale
│   ├── database.py          # connexion PostgreSQL
│   ├── models.py            # tables SQLAlchemy
│   ├── api/                 # routes (upload, analyse, rapport, chat, feedback)
│   ├── schemas/             # modèles Pydantic des réponses
│   └── services/            # logique métier
│       ├── preprocessing.py # prétraitement
│       ├── supervised.py    # modèles supervisés
│       ├── unsupervised.py  # clustering
│       ├── metrics.py       # métriques
│       ├── scoring.py       # score 0-100
│       ├── charts.py        # graphiques Plotly
│       ├── report.py        # rapport PDF
│       ├── notebook.py      # notebook Jupyter
│       └── rag/             # assistant (recherche, prompts, LLM, quotas)
├── data/
│   ├── knowledge/           # documents de la base de connaissances (Markdown)
│   ├── eval/questions.json  # jeu de test du RAG
│   └── feedback.jsonl       # votes des utilisateurs (créé automatiquement)
├── scripts/
│   ├── build_index.py       # reconstruit l'index vectoriel
│   ├── test_retrieval.py    # teste la recherche
│   ├── test_answer.py       # teste la chaîne complète
│   └── evaluate_rag.py      # évaluation chiffrée du RAG
├── uploads/                 # fichiers uploadés (créé automatiquement)
├── chroma_db/               # index vectoriel (créé par build_index.py)
├── requirements.txt
└── .env
```

## Prérequis

- Python 3.11 ou plus récent
- PostgreSQL 14 ou plus récent
- Une clé API Groq (gratuite) pour l'assistant : https://console.groq.com

## Installation

### 1. Créer l'environnement virtuel

```powershell
cd D:\DataAnalyzer\backend
python -m venv venv
venv\Scripts\activate
```

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
```

PyTorch est installé avec `sentence-transformers`. Il est volumineux (plusieurs centaines de Mo) et son installation peut prendre plusieurs minutes. Si vous voulez une version adaptée à votre machine (GPU, par exemple), suivez les instructions sur https://pytorch.org avant d'installer le reste.

### 3. Créer la base PostgreSQL

Dans pgAdmin ou psql :

```sql
CREATE DATABASE "DataAnalyzer";
```

### 4. Configurer les variables d'environnement

Créez le fichier `backend/.env` :

```
DATABASE_URL=postgresql+psycopg2://postgres:VOTRE_MOT_DE_PASSE@localhost:5432/DataAnalyzer
GROQ_API_KEY=votre_clé_groq
MAX_FILE_SIZE_MB=20
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Le fichier `.env` ne doit jamais être versionné.

### 5. Construire l'index de l'assistant

Une seule fois, puis à chaque modification des documents dans `data/knowledge/` :

```powershell
python scripts/build_index.py
```

Le premier lancement télécharge le modèle d'embeddings (quelques centaines de Mo).

### 6. Lancer le serveur

```powershell
uvicorn app.main:app --reload
```

Les tables PostgreSQL sont créées automatiquement au démarrage. Si vous ajoutez une colonne à un modèle existant, `create_all` ne la crée pas : il faut une commande `ALTER TABLE` ou une migration Alembic.

## Utilisation

- Documentation interactive : http://127.0.0.1:8000/docs
- Test rapide : `GET /health` doit répondre `{"status": "ok"}`

### Principales routes

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/api/upload` | Envoie un dataset (champ `file`) |
| GET | `/api/datasets/{id}/preview` | Aperçu du dataset |
| GET | `/api/datasets/{id}/charts` | Graphiques d'exploration |
| GET | `/api/datasets/{id}/detect` | Type de problème proposé (`target`, `problem_type` optionnels) |
| POST | `/api/analyze` | Lance l'analyse complète |
| POST | `/api/report` | Génère le rapport PDF |
| POST | `/api/notebook` | Génère le notebook Jupyter |
| GET | `/api/history` | Liste des analyses enregistrées |
| POST | `/api/chat` | Question à l'assistant |
| POST | `/api/chat/stream` | Réponse de l'assistant en flux |
| POST | `/api/feedback` | Vote sur une réponse de l'assistant |

### Exemple d'analyse

```json
{
  "dataset_id": "votre_id_retourné_par_upload",
  "target": "Species",
  "problem_type": null,
  "algorithms": null,
  "n_clusters": null
}
```

Pour du clustering, mettez `"target": null`.

## Évaluer l'assistant

```powershell
python scripts/evaluate_rag.py
```

Le script lit `data/eval/questions.json` et mesure le rappel de la recherche, le routage entre base de connaissances et LLM général, et la validité des citations. Il appelle le LLM pour chaque question : avec le quota gratuit, lancez-le en plusieurs fois si nécessaire.

## Limites connues

- Les quotas de l'assistant et les compteurs de requêtes sont en mémoire : ils sont remis à zéro au redémarrage.
- Le clustering hiérarchique et Mean Shift sont limités à 5 000 lignes, le SVM à 5 000 lignes d'entraînement.
- Les graphiques du rapport PDF ne sont pas inclus : seules les tables sont exportées.