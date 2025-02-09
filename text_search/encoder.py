import streamlit as st
import google.generativeai as genai


# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}


embedding_cache = get_embedding_cache()


def emb_text(text: str, model: str = "models/text-embedding-004"):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = genai.embed_content(
            model=model, content=text
        )["embedding"]
        embedding_cache[text] = embedding
        return embedding