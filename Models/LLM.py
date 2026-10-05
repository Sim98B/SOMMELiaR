import json
import logging
from typing import Iterator, List

import requests

from config import OPENROUTER_API_KEY

logger = logging.getLogger("LLM")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)


def generate_response_openrouter(
        model_name: str,
        messages: List[dict],
) -> Iterator[str]:
    """
    Args:
        model_name: modello di linguaggio da utilizzare.
        messages: lista di messaggi nel formato previsto dall'API OpenRouter.

    Yields:
        Frammenti della risposta generata dal modello.
    """

    logger.info(
        "Generazione risposta | "
        "model=%s | "
        "messages=%d",
        model_name,
        len(messages)
    )

    if not messages:
        logger.warning("Lista messaggi vuota, nessuna risposta generata")
        return

    logger.debug(
        "Statistiche input LLM | "
        "messaggi=%d",
        len(messages)
    )

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model_name,
            "messages": messages,
            "reasoning": {
                "enabled": False
            },
            "stream": True
        }),
        stream=True
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue

        data = line[6:]
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        content = chunk["choices"][0]["delta"].get("content")

        if content:
            yield content