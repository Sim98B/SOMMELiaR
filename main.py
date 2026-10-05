import logging

from config import (
    LOCAL_EMBEDDING_MODEL,
    OPENROUTER_LLM_MODEL,
)
from Models.LLM import generate_response_openrouter
from chroma import VectorStore


logger = logging.getLogger("MAIN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)


def main():

    vector_store = VectorStore()

    query = input("\n🍷 Tu: ")

    retrieved_results = vector_store.retrieve(
        query=query,
        embedding_model=LOCAL_EMBEDDING_MODEL,
        top_k=5
    )

    context = "\n\n".join(
        result["text"]
        for result in retrieved_results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Sei un sommelier esperto e appassionato. "
                "Devi guidare ed accompagnare gli utenti nella degustazione e alla scoperta dei vini"
                "Rispondi in modo naturale, chiaro e competente. "
                "Utilizza esclusivamente le informazioni presenti "
                "nel contesto fornito per rispondere alla domanda. "
                "Se il contesto non contiene informazioni sufficienti "
                "per rispondere, dichiaralo esplicitamente.\n\n"
                f"CONTESTO:\n{context}"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    print("\n🍷 SOMMELiaR: ", end="", flush=True)

    for chunk in generate_response_openrouter(
        model_name=OPENROUTER_LLM_MODEL,
        messages=messages
    ):
        print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":
    main()