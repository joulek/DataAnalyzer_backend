import os

# Ne remplace pas une valeur existante, mais corrige une valeur vide
if not os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@localhost:5432/test"