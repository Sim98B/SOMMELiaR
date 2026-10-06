from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from chroma import VectorStore
from config import LOCAL_EMBEDDING_MODEL, OPENROUTER_LLM_MODEL
from Models.LLM import generate_response_openrouter

from pathlib import Path
from fastapi.responses import FileResponse

from pydantic import BaseModel

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []

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
                "Devi guidare ed accompagnare gli utenti "
                "nella degustazione e alla scoperta dei vini. "
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

    return StreamingResponse(
        generate_response_openrouter(
            model_name=OPENROUTER_LLM_MODEL,
            messages=messages
        ),
        media_type="text/plain"
    )