import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OPENROUTER_URL = os.getenv("OPENROUTER_URL")
if not OPENROUTER_URL:
    raise RuntimeError("OPENROUTER_URL non configurata")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY non configurata")

OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL")
if not OPENROUTER_EMBEDDING_MODEL:
    raise RuntimeError("OPENROUTER_EMBEDDING_MODEL non configurata")

LLM_MODEL = os.getenv("LLM")
if not LLM_MODEL:
    raise RuntimeError("LLM_MODEL non configurata")

COLLECTION = os.getenv("COLLECTION_NAME")
if not COLLECTION:
    raise RuntimeError("COLLECTION non configurata")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
if not COLLECTION:
    raise RuntimeError("MAX_RETRIES non configurata")

RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
if not COLLECTION:
    raise RuntimeError("RETRY_DELAY non configurata")