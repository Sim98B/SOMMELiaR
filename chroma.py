import logging
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from Templates import VectorChunk
from Models.Embed import generate_embedding, stop_ollama_model
from Chunkers import chunk_core, chunk_tasting, chunk_pairing

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

    def retrieve(
            self,
            query,
            embedding_model,
            embedding_dimension,
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
            dim=embedding_dimension
        )[0]
        stop_ollama_model(embedding_model)

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

def prepare_metadata(metadata: dict):
    return {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, dict))
        else value
        for key, value in metadata.items()
    }

def insert_annotations(
        md_file_path: str,
        chunker: str,
        collection,
        embedding_model: str,
        embedding_dim: int
):
    """
    Estrae i chunk, genera gli embedding e li inserisce in Chroma.

    Args:
        md_file_path: file markdown del vino
        chunker: tipo di chunk da creare
        embedding_model: modello embedding Ollama
        collection: collection Chroma
    """

    logger.info(f"Inizio inserimento annotazioni")

    logger.debug("File: %s", md_file_path)
    logger.debug("Section: %s", chunker)
    logger.debug("Embedding model: %s", embedding_model)

    chunkers = {
        "core": chunk_core,
        "tasting": chunk_tasting,
        "pairing": chunk_pairing
    }

    if chunker not in chunkers:
        logger.error(f"Chunker non valido: {chunker}. " f"Disponibili: {list(chunkers.keys())}")
        raise ValueError(f"Chunker '{chunker}' non disponibile. " f"Scegli tra {list(chunkers.keys())}")

    try:
        chunks = chunkers[chunker](md_file_path)

    except Exception:
        logger.exception(f"Errore durante creazione chunk | file={md_file_path}")
        raise

    if isinstance(chunks, VectorChunk):
        chunks = [chunks]
    logger.info(f"Creati {len(chunks)} chunk | " f"tipo={chunker}")

    if len(chunks) == 0:
        logger.warning(f"Nessun chunk generato per il file {md_file_path}")
        return

    texts = [chunk.text for chunk in chunks]
    logger.info(f"Generazione embedding per {len(texts)} testi")

    try:
        embeddings = generate_embedding(
            model_name=embedding_model,
            dim=embedding_dim,
            text=texts
        )

    except Exception:
        logger.exception("Errore durante generazione embedding")
        raise

    logger.info(
        f"Embedding generati correttamente | "
        f"numero={len(embeddings)} | "
        f"dimensione={len(embeddings[0]) if embeddings else 0}"
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    try:
        collection.upsert(
            ids=[
                chunk.id
                for chunk in chunks
            ],
            documents=[
                chunk.text
                for chunk in chunks
            ],
            embeddings=[
                chunk.embedding
                for chunk in chunks
            ],
            metadatas = [
                prepare_metadata(chunk.metadata)
                for chunk in chunks
            ]
        )

    except Exception:
        logger.exception(f"Errore inserimento ChromaDB | " f"chunk inseriti={len(chunks)}")
        raise

    logger.info(f"Inserimento completato | " f"chunk aggiunti={len(chunks)} | " f"collection_size={collection.count()}")

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