import streamlit as st

from config import (
    LOCAL_EMBEDDING_MODEL,
    OPENROUTER_LLM_MODEL,
)
from Models.LLM import generate_response_openrouter
from chroma import VectorStore


st.set_page_config(
    page_title="SOMMELiaR",
    page_icon="🍷",
    layout="centered",
)


@st.cache_resource
def load_vector_store():
    return VectorStore()


vector_store = load_vector_store()


st.markdown(
    """
    <style>

    .stApp {
        background-color: #1c1512;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #b9aaa3;
        font-size: 16px;
        margin-bottom: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🍷 SOMMELiaR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Il tuo sommelier personale</div>',
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


query = st.chat_input(
    "Chiedimi qualcosa sul vino..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

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

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        response = ""

        for chunk in generate_response_openrouter(
            model_name=OPENROUTER_LLM_MODEL,
            messages=messages
        ):
            response += chunk
            response_placeholder.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )