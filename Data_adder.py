import json
import logging
from pathlib import Path
from chroma import VectorStore, plot_chroma_embeddings_3d
from Templates import Wine
from Models.Embed import generate_embedding_local
from config import LOCAL_EMBEDDING_MODEL

logger = logging.getLogger("ADDER")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

WINES_PATH = Path("Wines/Data.json")
#WINES_PATH = Path("Wines/Data.json")
SECTIONS = [
    "profilo",
    "caratteristiche",
    "servizio",
    "esame_sensoriale",
    "esame_gustativo",
    "abbinamenti",
]
with open(WINES_PATH, "r", encoding="utf-8") as f:
    wines = json.load(f)

logger.info("Vini caricati: %d",len(wines))

db = VectorStore()

for wine_name, sections in wines.items():
    logger.info("Elaborazione vino: %s",wine_name)

    chunks = []
    for section in SECTIONS:
        text = sections.get(section, "")

        if not text:
            logger.warning("Campo vuoto | vino=%s | sezione=%s",wine_name,section)
            continue

        chunks.append(
            Wine(id=f"{wine_name.replace(" ","_").lower()}_{section}", text=text, embedding=[])
        )

    if not chunks:
        logger.warning("Nessun chunk trovato | vino=%s",wine_name)
        continue

    texts = [chunk.text for chunk in chunks]

    logger.info("Generazione embedding | vino=%s | chunks=%d",wine_name,len(texts))

    """embeddings = generate_embedding(
        api_key=OPENROUTER_API_KEY,
        model_name=OPENROUTER_EMBEDDING_MODEL,
        text=texts
    )"""

    embeddings = generate_embedding_local(model_name=LOCAL_EMBEDDING_MODEL, text=texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Numero embedding inatteso | "
            f"vino={wine_name} | "
            f"chunks={len(chunks)} | "
            f"embeddings={len(embeddings)}"
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    db.collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=[chunk.embedding for chunk in chunks],
        metadatas=[
            {
                "wine_name": wine_name,
                "section": chunk.id.rsplit("_", 1)[-1]
            }
            for chunk in chunks
        ]
    )

    logger.info("Vino inserito | vino=%s | chunks=%d",wine_name,len(chunks))

logger.info("Inserimento completato | collection_size=%d", db.collection.count())

plot_chroma_embeddings_3d(
    collection=db.collection,
    n_components=3
)