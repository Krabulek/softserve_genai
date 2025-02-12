import streamlit as st
import google.generativeai as genai
import torch
import sys
import os

# necessary for the FlagEmbedding import to work
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from FlagEmbedding.research.visual_bge.modeling import Visualized_BGE


class ImageEncoder:
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]
    
    def encode_text(self, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(text=text)
        return query_emb.tolist()[0]


model_name = "BAAI/bge-base-en-v1.5"
model_path = "./Visualized_base_en_v1.5.pth"
encoder = ImageEncoder(model_name, model_path)


# Cache for text embeddings
@st.cache_resource
def get_text_embedding_cache():
    return {}


# Cache for image embeddings
@st.cache_resource
def get_image_embedding_cache():
    return {}


text_embedding_cache = get_text_embedding_cache()
image_embedding_cache = get_image_embedding_cache()

def emb_text(text: str, model: str = "models/text-embedding-004"):
    if text in text_embedding_cache:
        return text_embedding_cache[text]
    embedding = genai.embed_content(
        model=model, content=text
    )["embedding"]
    text_embedding_cache[text] = embedding
    return embedding
    

def emb_image(image_path: str):
    if image_path in image_embedding_cache:
        return image_embedding_cache[image_path]
    try:
        embedding = encoder.encode_image(image_path)
        image_embedding_cache[image_path] = embedding
        return embedding
    except Exception as e:
        print(f"Failed to generate embedding for {image_path}. Skipped.")


def emb_image_text(text: str):
    try:
        embedding = encoder.encode_text(text)
        return embedding
    except Exception as e:
        print(f"Failed to generate embedding for {text}. Skipped.")
