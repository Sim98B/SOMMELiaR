from chroma import VectorStore
from config import LOCAL_EMBEDDING_MODEL
import json

def ask_sommelier(query: str):

    db = VectorStore()

    vector_results = db.retrieve(
        query=query,
        embedding_model=LOCAL_EMBEDDING_MODEL,
        top_k=7
    )

    return vector_results