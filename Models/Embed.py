import json
import logging
import time
from typing import List

import requests

from config import OPENROUTER_URL, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger("EMBED")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

def generate_embedding(
        api_key: str,
        model_name: str,
        text: List[str],
) -> List[List[float]]:
    """
    Genera gli embedding per una lista di stringhe.
    Args:
        api_key: chiave API OpenRouter.
        model_name: modello di embedding da utilizzare.
        text: lista di testi da trasformare in embedding.
    Returns:
        Lista di embedding.
    Raises:
        ValueError: se la risposta API non contiene dati validi.
        RuntimeError: se OpenRouter restituisce un errore.
        requests.RequestException: se si verifica un errore HTTP/network.
    """

    logger.info(
        "Generazione embedding | "
        "model=%s | "
        "chunks=%d",
        model_name,
        len(text)
    )

    if not text:
        logger.warning("Lista testi vuota, nessun embedding generato")
        return []

    avg_length = sum(len(t) for t in text) / len(text)

    logger.debug(
        "Statistiche input embedding | "
        "lunghezza_media_testo=%.0f caratteri | "
        "max_length=%d",
        avg_length,
        max(len(t) for t in text)
    )

    payload = {
        "model": model_name,
        "input": text,
        "encoding_format": "float"
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "SOMMELiaR",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.time()
        try:
            response = requests.post(
                url=OPENROUTER_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )

        except requests.RequestException as e:
            logger.error(
                "Errore connessione OpenRouter | "
                "tentativo=%d/%d | "
                "errore=%s",
                attempt,
                MAX_RETRIES,
                e
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(RETRY_DELAY * attempt)
            continue

        elapsed = time.time() - start_time

        if not response.ok:
            try:
                error_data = response.json()
            except ValueError:
                error_data = {"message": response.text}

            error = error_data.get("error", {})

            error_code = error.get("code", response.status_code)

            error_message = error.get("message", "Errore sconosciuto")

            logger.error(
                "Errore OpenRouter | "
                "status=%d | "
                "code=%s | "
                "message=%s",
                response.status_code,
                error_code,
                error_message
            )

            if response.status_code == 429:
                logger.warning("Rate limit OpenRouter raggiunto")

                if attempt < MAX_RETRIES:
                    retry_after = response.headers.get("Retry-After")

                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = RETRY_DELAY * attempt
                    else:
                        delay = RETRY_DELAY * attempt

                    logger.info("Nuovo tentativo tra %.1f secondi",delay)
                    time.sleep(delay)
                    continue

                raise RuntimeError(f"OpenRouter rate limit exceeded: " f"{error_message}")

            if response.status_code >= 500:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * attempt
                    logger.warning(
                        "Errore server OpenRouter (%d). "
                        "Nuovo tentativo tra %.1f secondi",
                        response.status_code,
                        delay
                    )

                    time.sleep(delay)
                    continue

                raise RuntimeError(f"OpenRouter server error " f"{response.status_code}: " f"{error_message}")
            raise RuntimeError(f"OpenRouter API error " f"{response.status_code}: " f"{error_message}")

        try:
            result = response.json()
        except ValueError as e:
            raise RuntimeError("OpenRouter ha restituito una risposta " "JSON non valida") from e

        if "data" not in result:
            raise ValueError("Risposta OpenRouter priva del campo 'data'")

        embeddings = [item["embedding"] for item in result["data"]]

        if len(embeddings) != len(text):
            raise ValueError("Numero embedding inatteso | " f"richiesti={len(text)} | " f"ricevuti={len(embeddings)}")

        logger.info(
            "Embedding generati | "
            "numero=%d | "
            "dimensione=%d | "
            "tempo=%.2fs",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
            elapsed
        )

        return embeddings

    raise RuntimeError("Impossibile generare gli embedding")