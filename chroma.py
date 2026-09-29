import logging
from pathlib import Path
import chromadb
from Models.Embed import generate_embedding

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

logger = logging.getLogger("CHROMA")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

class VectorStore:
    def __init__(self, path=CHROMA_PATH):
        logger.info("Inizializzazione VectorStore...")
        self.path = path
        try:
            self.client = chromadb.PersistentClient(path=str(path))
            logger.info(f"Client ChromaDB creato con path: {path}")
            self.collection = self.client.get_or_create_collection(name="wine_knowledge")
            logger.info(f"Collection caricata: {self.collection.name}")
            logger.info(f"Numero elementi presenti nella collection: " f"{self.collection.count()}")

        except Exception as e:
            logger.exception(f"Errore durante inizializzazione VectorStore: {e}")
            raise

def plot_chroma_embeddings_3d(
        collection,
        n_components=3,
        title="ChromaDB Embeddings 3D"
):
    """
    Visualizza gli embedding presenti in una collection Chroma
    ridotti a 3 dimensioni con PCA.

    Args:
        collection: collection ChromaDB
        n_components: numero dimensioni PCA
        title: titolo grafico
    """

    # Recupero dati dal DB
    data = collection.get(
        include=[
            "embeddings",
            "documents",
            "metadatas"
        ]
    )

    embeddings = np.array(data["embeddings"])

    if len(embeddings) < 3:
        raise ValueError(
            "Servono almeno 3 embedding per una visualizzazione 3D"
        )

    print(f"Embedding caricati: {embeddings.shape}")


    # Riduzione dimensionale
    pca = PCA(
        n_components=n_components
    )

    reduced = pca.fit_transform(embeddings)


    # Dataframe per Plotly
    df = pd.DataFrame(
        reduced,
        columns=[
            "PC1",
            "PC2",
            "PC3"
        ]
    )


    # Aggiungo informazioni utili
    df["id"] = data["ids"]

    df["text"] = [
        doc[:100] + "..."
        if doc and len(doc) > 100
        else doc
        for doc in data["documents"]
    ]

    df["wine"] = [
        meta.get("wine_name", "unknown")
        if meta
        else "unknown"
        for meta in data["metadatas"]
    ]


    # Plot 3D
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="wine",
        hover_data=[
            "id",
            "text"
        ],
        title=title
    )

    fig.update_layout(
        width=1000,
        height=800
    )

    fig.show()

    return df

def retrieve(
        self,
        query,
        embedding_model,
        top_k=5
):
    """
    Ricerca semantica tramite embedding + ChromaDB.

    Args:
        query: query testuale dell'utente
        embedding_model: nome del modello Ollama utilizzato per gli embedding
        top_k: numero massimo di risultati da restituire

    Returns:
        Lista di risultati nello stesso formato di BM25Retriever.retrieve()
    """

    # Generazione embedding della query
    query_embedding = generate_embedding(
        model_name=embedding_model,
        text=[query],
    )[0]
    # Ricerca semantica in ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_results = []

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]

        retrieved_results.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": distance
        })

    return retrieved_results