from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from chroma import VectorStore
from config import LOCAL_EMBEDDING_MODEL, OPENROUTER_LLM_MODEL
from Models.LLM import generate_response_openrouter

from pathlib import Path
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)

vector_store = VectorStore()


@app.get("/")
def root():
    return {"status": "SOMMELiaR online"}

@app.get("/app")
def app_page():
    frontend_path = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "index.html"
    )

    return FileResponse(frontend_path)


@app.get("/search")
def search(query: str):
    results = vector_store.retrieve(
        query=query,
        embedding_model=LOCAL_EMBEDDING_MODEL,
        top_k=5
    )

    return {
        "query": query,
        "results": [
            {
                "id": result["id"],
                "text": result["text"],
                "metadata": result["metadata"],
                "score": result["score"]
            }
            for result in results
        ]
    }


@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query
    history = request.history[-6:]

    # Costruisce il contesto della conversazione precedente
    history_context = "\n".join(
        f"{message.role}: {message.content}"
        for message in history
    )

    # Query utilizzata esclusivamente per il retrieval
    if history_context:
        retrieval_query = (
            f"Conversazione precedente:\n"
            f"{history_context}\n\n"
            f"Nuova domanda:\n"
            f"{query}"
        )
    else:
        retrieval_query = query

    retrieved_results = vector_store.retrieve(
        query=retrieval_query,
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
                "Devi guidare ed accompagnare gli utenti "
                "nella degustazione e alla scoperta dei vini. "
                "Rispondi in modo naturale, chiaro e competente.\n\n"

                "Utilizza esclusivamente le informazioni presenti "
                "nel contesto fornito per rispondere alla domanda. "
                "Non utilizzare conoscenze generali o informazioni "
                "apprese durante l'addestramento per integrare, "
                "completare o arricchire il contesto.\n\n"

                "Se il contesto non contiene informazioni sufficienti "
                "per rispondere, dichiaralo semplicemente. "
                "Non fornire comunque una risposta basata sulla tua "
                "conoscenza generale.\n\n"

                "La conversazione precedente serve per comprendere "
                "il riferimento delle domande dell'utente. "
                "Per esempio, se l'utente ha parlato del Brunello "
                "di Montalcino e successivamente chiede "
                "\"E con cosa lo abbineresti?\", interpreta \"lo\" "
                "come riferito al Brunello di Montalcino.\n\n"

                f"CONTESTO:\n{context}"
            )
        }
    ]

    messages.extend(
        {
            "role": message.role,
            "content": message.content
        }
        for message in history
    )

    messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    return StreamingResponse(
        generate_response_openrouter(
            model_name=OPENROUTER_LLM_MODEL,
            messages=messages
        ),
        media_type="text/plain"
    )