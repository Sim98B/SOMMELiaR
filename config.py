import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY non configurata")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
if not EMBEDDING_MODEL:
    raise RuntimeError("EMBEDDING_MODEL non configurata")

LLM_MODEL = os.getenv("LLM")
if not LLM_MODEL:
    raise RuntimeError("LLM_MODEL non configurata")

COLLECTION = os.getenv("COLLECTION_NAME")
if not COLLECTION:
    raise RuntimeError("COLLECTION non configurata")