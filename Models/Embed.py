import requests
import json
import logging
import time
from typing import List

logger = logging.getLogger("EMBED")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

def generate_embedding(
        api_key: str,
        model_name: str,
        text: List[str],
) -> List[float]:
    """
    Genera gli embedding per una lista di stringhe
    :param api_key: chiave OpenRouter per l'utilizzo dei modelli
    :param model_name: nome del modello di embedding da usare
    :param text: lista di stringhe da trasformare in embedding
    :return: lista di embeddings
    """

    logger.info(f"Generazione embedding | " f"model={model_name} | " f"chunks={len(text)}")
    if not text:
        logger.warning("Lista testi vuota, nessun embedding generato")
        return []

    avg_length = sum(len(t) for t in text) / len(text)
    logger.debug(
        f"Statistiche input embedding | "
        f"lunghezza_media_testo={avg_length:.0f} caratteri | "
        f"max_length={max(len(t) for t in text)}"
    )

    start_time = time.time()

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "SOMMELiaR",
            },
            data=json.dumps({
                "model": model_name,
                "input": text,
                "encoding_format": "float"
            })
        )
    except Exception:
        logger.exception(f"Errore generazione embedding | " f"model={model_name}")
        raise

    elapsed = time.time() - start_time

    result = response.json()
    embeddings = [
        item["embedding"]
        for item in result["data"]
    ]
    logger.info(
        f"Embedding generati | "
        f"numero={len(embeddings)} | "
        f"dimensione={len(embeddings[0]) if embeddings else 0} | "
        f"tempo={elapsed:.2f}s"
    )

    return embeddings