import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY non configurata")

print(f"OPENROUTER_API_KEY: {OPENROUTER_API_KEY}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"LLM_MODEL: {LLM_MODEL}")