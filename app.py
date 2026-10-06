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

    /* Sfondo */

    .stApp {
        background-color: #f3eadc;
        color: #000000;
    }


    /* Nasconde header e footer di Streamlit */

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Area principale */

    .block-container {
        max-width: 850px;
        padding-top: 0;
        padding-bottom: 120px;
    }


    /* Testo generale */

    p {
        color: #000000;
    }


    /* Input */

    .stChatInput {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: min(700px, 90%);
    }


    .stChatInput > div {
        background-color: #ffffff;
        border: 1px solid #b8aa99;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }


    .stChatInput textarea {
        color: #000000 !important;
        background-color: transparent !important;
    }


    .stChatInput textarea::placeholder {
        color: #77716a !important;
    }


    /* Messaggi */

    [data-testid="stChatMessage"] {
        background-color: transparent;
        color: #000000;
        border: none;
    }


    [data-testid="stChatMessage"] p {
        color: #000000;
    }


    /* Nasconde le icone/avatar dei messaggi */

    [data-testid="stChatMessageAvatar"] {
        display: none;
    }


    </style>
    """,
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