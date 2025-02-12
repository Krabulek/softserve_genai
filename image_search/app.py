import os
import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text, emb_image_text
from milvus_utils import get_milvus_client, get_search_text_results, get_search_image_results
from ask_llm import get_llm_answer
import google.generativeai as genai

from dotenv import load_dotenv


load_dotenv(override=True)

TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME")
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-lite-preview-02-05"

genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the articles provided.
"""


milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
gemini_model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

st.logo("./brain.png", size='large')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">Multimodal RAG</div>
    <div class="description">
        This chatbot is built with Milvus vector database, utilising a multimodal 
        <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge/">Visualised-BGE</a> model
        and Gemini text embedding model.
        The system uses Gemini 2.0 Flash Lite as an LLM and supports conversation based on the articles from <a href="https://www.deeplearning.ai/the-batch/">The Batch</a> website.
    </div>
    """,
    unsafe_allow_html=True,
)


retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("Enter your question:")
    
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # Text part
        query_vector = emb_text(question)
        search_res = get_search_text_results(
            milvus_client, TEXT_COLLECTION_NAME, query_vector, ["article_url", "image_url", "text"]
        )

        retrieved_lines_with_distances = [
            (
                res["distance"],
                res["entity"]["article_url"],
                res["entity"]["image_url"],
                res["entity"]["text"].replace(u'\u2019', u'\'')
            ) for res in search_res[0]
        ]

        context = "\n".join([line_with_distance[3] for line_with_distance in retrieved_lines_with_distances])
        answer = get_llm_answer(gemini_model, context, question)

        # Image Part
        img_query_vector = emb_image_text(question)
        images_retrieved = get_search_image_results(milvus_client, IMAGE_COLLECTION_NAME, img_query_vector)

        # Final
        final_answer = f"Gemini: {answer}\n\nSimilar images:"

        st.chat_message("user").write(question)
        st.chat_message("assistant").write(final_answer)
        
        st.image(images_retrieved, caption=images_retrieved)


st.sidebar.subheader("Retrieved Articles:")
for idx, (distance, article_url, image_url, text) in enumerate(retrieved_lines_with_distances, 1):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"*Article: {article_url}*")
    st.sidebar.markdown(f"*Image: {image_url}*")
    st.sidebar.markdown(f"*Distance: {distance:.2f}*")
    st.sidebar.markdown(f"**Result {idx}:**")
    st.sidebar.markdown(f"> {text[:min(len(text), 300)]}")