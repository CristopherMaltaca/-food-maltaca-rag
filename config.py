import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CARDAPIO_PATH = BASE_DIR / "cardapio.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"
HASH_PATH = BASE_DIR / ".index_hash"

COLLECTION_NAME = "cardapio_food_maltaca"
K_RESULTADOS = 6

MODELO = "gemini-3.6-flash"
MODELOS_FALLBACK = ["gemini-3.5-flash-lite"]
THINKING_LEVEL = "low"
TEMPERATURA = 0.2
TIMEOUT_MS = 40000

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
