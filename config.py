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

OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM")
if not OPENROUTER_LLM_MODEL:
    raise RuntimeError("OPENROUTER_LLM_MODEL non configurata")

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL")
if not OPENROUTER_EMBEDDING_MODEL:
    raise RuntimeError("LOCAL_EMBEDDING_MODEL non configurata")

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM")
if not LOCAL_LLM_MODEL:
    raise RuntimeError("LOCAL_LLM_MODEL non configurata")

COLLECTION = os.getenv("COLLECTION_NAME")
if not COLLECTION:
    raise RuntimeError("COLLECTION non configurata")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
if not COLLECTION:
    raise RuntimeError("MAX_RETRIES non configurata")

RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
if not COLLECTION:
    raise RuntimeError("RETRY_DELAY non configurata")