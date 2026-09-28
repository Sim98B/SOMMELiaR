import logging
from pathlib import Path
from typing import List
import re
from Templates import VectorChunk

logger = logging.getLogger("CHUNKERS")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

def chunk_core(file_path: str) -> VectorChunk:
    """
    Legge un file markdown di un vino e crea il VectorChunk core.

    Il testo della sezione Overview viene usato come testo del chunk.
    Le informazioni strutturate delle sezioni Informazioni e Denominazioni
    vengono salvate come metadata flat.

    Args:
        file_path: percorso del file markdown

    Returns:
        VectorChunk senza embedding
    """

    logger.info(f"Creazione chunk core | file={file_path}")

    try:
        markdown = Path(file_path).read_text(encoding="utf-8")

    except Exception:
        logger.exception(
            f"Errore lettura file markdown | file={file_path}"
        )
        raise

    logger.debug(
        f"File letto correttamente | caratteri={len(markdown)}"
    )

    def extract_section(md: str, section_name: str):
        pattern = rf"## {section_name}\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, md, flags=re.S)

        if match:
            return match.group(1).strip()

        return None

    overview_text = extract_section(markdown, "Overview")
    info_text = extract_section(markdown, "Informazioni")
    denominations_text = extract_section(markdown, "Denominazioni")

    logger.debug(
        f"Sezioni estratte | "
        f"Overview={'presente' if overview_text else 'mancante'} | "
        f"Informazioni={'presente' if info_text else 'mancante'} | "
        f"Denominazioni={'presente' if denominations_text else 'mancante'}"
    )

    if overview_text is None:
        logger.error(
            f"Sezione Overview mancante | file={file_path}"
        )
        raise ValueError("Sezione Overview mancante")

    wine_name = (
        Path(file_path)
        .stem
        .capitalize()
        .replace("_", " ")
    )

    wine_id = wine_name.lower().replace(" ", "_")

    logger.info(
        f"Parsing vino | nome={wine_name} | wine_id={wine_id}"
    )

    def extract_list_field(text: str, field: str) -> List[str]:
        if not text:
            return []

        pattern = rf"\*\*{field}\*\*\n((?:- .+\n?)+)"

        match = re.search(pattern, text)

        if not match:
            return []

        values = match.group(1).strip().split("\n")

        return [
            v.replace("- ", "").strip()
            for v in values
        ]

    colore = extract_list_field(
        info_text,
        "Colore"
    )

    tipologie = extract_list_field(
        info_text,
        "Tipologie"
    )

    vitigni = extract_list_field(
        info_text,
        "Vitigni"
    )

    regioni = extract_list_field(
        info_text,
        "Regione"
    )

    denominazioni = [
        x.replace("- ", "").strip()
        for x in denominations_text.split("\n")
        if x.strip().startswith("-")
    ] if denominations_text else []

    metadata = {
        "wine_id": wine_id,
        "wine_name": wine_name,
        "type": "core",
        "colore": " ".join(colore).lower(),
        "tipologie": " ".join(tipologie).lower(),
        "vitigni": " ".join(vitigni).lower(),
        "regioni": " ".join(regioni).lower(),
        "denominazioni": " ".join(denominazioni).lower()
    }

    vector_chunk = VectorChunk(
        id=f"{wine_id}_core",
        text=overview_text,
        metadata=metadata,
        embedding=None
    )

    logger.info(
        f"Chunk core creato | "
        f"id={vector_chunk.id} | "
        f"lunghezza_testo={len(vector_chunk.text)}"
    )

    return vector_chunk

def chunk_tasting(file_path: str) -> VectorChunk | None:
    """
    Estrae la sezione Degustazione e crea un singolo VectorChunk
    di tipo tasting.

    Il testo completo e discorsivo della degustazione viene mantenuto
    nel campo `text`. Le proprietà e i descrittori vengono estratti
    separatamente nei metadata.
    """

    logger.info(f"Creazione chunk tasting | file={file_path}")

    try:
        markdown = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        logger.exception(
            f"Errore lettura file markdown | file={file_path}"
        )
        raise

    # --------------------------------------------------------------
    # Informazioni vino
    # --------------------------------------------------------------

    wine_name = (
        Path(file_path)
        .stem
        .capitalize()
        .replace("_", " ")
    )

    wine_id = wine_name.lower().replace(" ", "_")

    # --------------------------------------------------------------
    # Estrazione Degustazione
    # --------------------------------------------------------------

    tasting_match = re.search(
        r"^## Degustazione\s*\n(.*?)(?=^## |\Z)",
        markdown,
        flags=re.S | re.M
    )

    if not tasting_match:
        logger.warning(
            f"Sezione Degustazione assente | file={file_path}"
        )
        return None

    tasting_text = tasting_match.group(1).strip()

    logger.debug(
        f"Degustazione estratta | "
        f"lunghezza={len(tasting_text)}"
    )

    # --------------------------------------------------------------
    # Funzione generica per estrarre una sottosezione
    # --------------------------------------------------------------

    def extract_subsection(
        parent_section: str,
        subsection: str
    ) -> str:

        pattern = (
            rf"^### {re.escape(parent_section)}\s*\n"
            rf"(.*?)"
            rf"(?=^### |\Z)"
        )

        parent_match = re.search(
            pattern,
            tasting_text,
            flags=re.S | re.M
        )

        if not parent_match:
            return ""

        parent_text = parent_match.group(1)

        pattern = (
            rf"^#### {re.escape(subsection)}\s*\n"
            rf"(.*?)"
            rf"(?=^#### |\Z)"
        )

        match = re.search(
            pattern,
            parent_text,
            flags=re.S | re.M
        )

        if not match:
            return ""

        return match.group(1).strip()

    # --------------------------------------------------------------
    # Proprietà
    # --------------------------------------------------------------

    properties_text = extract_subsection(
        "Caratteristiche",
        "Proprietà"
    )

    def extract_property(name: str) -> str:

        if not properties_text:
            return ""

        pattern = (
            rf"^\s*-\s*\*\*{re.escape(name)}\*\*:\s*"
            rf"(.*?)"
            rf"(?=^\s*-\s*\*\*|\Z)"
        )

        match = re.search(
            pattern,
            properties_text,
            flags=re.S | re.M
        )

        if not match:
            return ""

        value = match.group(1).strip()

        values = re.findall(
            r"^\s*-\s*(.+)$",
            value,
            flags=re.M
        )

        if values:
            return " ".join(
                value.strip().lower()
                for value in values
            )

        return value.lower()

    tipologie = extract_property("Tipologie")
    struttura = extract_property("Struttura")
    qualita = extract_property("Qualità")
    temperatura_servizio = extract_property(
        "Temperatura di servizio"
    )
    bicchiere = extract_property("Bicchiere")

    # --------------------------------------------------------------
    # Descrittori
    # --------------------------------------------------------------

    def extract_descriptors(section_name: str) -> str:

        descriptors_text = extract_subsection(
            section_name,
            "Descrittori"
        )

        if not descriptors_text:
            return ""

        descriptors = re.findall(
            r"^\s*-\s*(.+)$",
            descriptors_text,
            flags=re.M
        )

        return " ".join(
            descriptor.strip().lower()
            for descriptor in descriptors
        )

    descrittori_visivi = extract_descriptors(
        "Esame visivo"
    )

    descrittori_olfattivi = extract_descriptors(
        "Esame olfattivo"
    )

    descrittori_gustativi = extract_descriptors(
        "Esame gusto-olfattivo"
    )

    # --------------------------------------------------------------
    # Pulizia testo per embedding
    # --------------------------------------------------------------

    embedding_text = tasting_text

    # Rimuove gli heading Markdown ma NON il loro contenuto
    embedding_text = re.sub(
        r"^#{3,4}\s+",
        "",
        embedding_text,
        flags=re.M
    )

    # Rimuove il markup bold
    embedding_text = embedding_text.replace("**", "")

    # Normalizza spaziature
    embedding_text = re.sub(
        r"\n{3,}",
        "\n\n",
        embedding_text
    ).strip()

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata = {
        "wine_id": wine_id,
        "wine_name": wine_name,
        "type": "tasting",

        "tipologie": tipologie,
        "struttura": struttura,
        "qualita": qualita,
        "temperatura_servizio": temperatura_servizio,
        "bicchiere": bicchiere,

        "descrittori_visivi": descrittori_visivi,
        "descrittori_olfattivi": descrittori_olfattivi,
        "descrittori_gustativi": descrittori_gustativi
    }

    # --------------------------------------------------------------
    # VectorChunk
    # --------------------------------------------------------------

    vector_chunk = VectorChunk(
        id=f"{wine_id}_tasting",
        text=embedding_text,
        metadata=metadata,
        embedding=None
    )

    logger.info(
        f"Chunk tasting creato | "
        f"id={vector_chunk.id} | "
        f"lunghezza_originale={len(tasting_text)} | "
        f"lunghezza_embedding={len(embedding_text)}"
    )

    return vector_chunk

def chunk_pairing(file_path: str) -> VectorChunk | None:
    """
    Legge il file markdown e crea un VectorChunk della sezione Abbinamenti.

    Args:
        file_path: percorso del file markdown

    Returns:
        VectorChunk senza embedding oppure None se la sezione è assente.
    """

    logger.info(f"Creazione chunk pairing | file={file_path}")

    try:
        markdown = Path(file_path).read_text(encoding="utf-8")

    except Exception:
        logger.exception(f"Errore lettura file markdown | file={file_path}")
        raise

    logger.debug(
        f"File letto correttamente | caratteri={len(markdown)}"
    )

    # --------------------------------------------------
    # Identificazione vino
    # --------------------------------------------------

    wine_name = Path(file_path).stem.capitalize().replace("_", " ")
    wine_id = wine_name.lower().replace(" ", "_")

    logger.debug(
        f"Nome vino estratto | nome={wine_name} | wine_id={wine_id}"
    )

    # --------------------------------------------------
    # Estrazione sezione Abbinamenti
    # --------------------------------------------------

    pattern = r"## Abbinamenti\s*\n(.*?)(?=\n## |\Z)"

    match = re.search(
        pattern,
        markdown,
        flags=re.S
    )

    if not match:
        logger.warning(
            f"Sezione Abbinamenti assente | file={file_path}"
        )
        return None

    pairing_text = match.group(1).strip()

    logger.debug(
        f"Sezione Abbinamenti estratta | "
        f"lunghezza_testo={len(pairing_text)}"
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    metadata = {
        "wine_id": wine_id,
        "wine_name": wine_name,
        "type": "pairing"
    }

    # --------------------------------------------------
    # Creazione VectorChunk
    # --------------------------------------------------

    vector_chunk = VectorChunk(
        id=f"{wine_id}_pairing",
        text=pairing_text,
        metadata=metadata,
        embedding=None
    )

    logger.info(
        f"Chunk pairing creato | "
        f"id={vector_chunk.id} | "
        f"lunghezza_testo={len(vector_chunk.text)}"
    )

    return vector_chunk

#import json
#chunk = chunk_core("/Users/simonebattistini/Desktop/SOMMELiaR/Wines/AGLIANICO.md")
#chunk = chunk_pairing("/Users/simonebattistini/Desktop/SOMMELiaR/Wines/VIOGNIER.md")
#chunk = chunk_tasting("/Users/simonebattistini/Desktop/SOMMELiaR/Wines/ANSONICA_E_INZOLIA.md")
#print(json.dumps(chunk.model_dump(), indent=4, ensure_ascii=False))