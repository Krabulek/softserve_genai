import os
import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import get_llm_answer
import google.generativeai as genai

from dotenv import load_dotenv


load_dotenv(override=True)
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Logo
# st.image("./pics/Milvus_Logo_Official.png", width=200)

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
    <div class="title">RAG Demo</div>
    <div class="description">
        This chatbot is built with Milvus vector database, supported by Gemini text embedding model.<br>
        It supports conversation based on the articles from <a href="https://www.deeplearning.ai/the-batch/">The Batch</a> website.
    </div>
    """,
    unsafe_allow_html=True,
)

SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the articles provided.
"""

# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)
gemini_model = genai.GenerativeModel(
        "gemini-2.0-flash-lite-preview-02-05", system_instruction=SYSTEM_PROMPT
    )


retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("Enter your question:")
    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # Generate query embedding
        query_vector = emb_text(question)
        # Search in Milvus collection
        search_res = get_search_results(
            milvus_client, COLLECTION_NAME, query_vector, ["text", "article_url", "image_url"]
        )

        # Retrieve lines and distances
        retrieved_lines_with_distances = [
            (
                res["distance"],
                res["entity"]["article_url"],
                res["entity"]["image_url"],
                res["entity"]["text"].replace(u'\u2019', u'\'')
            ) for res in search_res[0]
        ]

        # Create context from retrieved lines
        context = "\n".join(
            [
                line_with_distance[3]
                for line_with_distance in retrieved_lines_with_distances
            ]
        )
        answer = get_llm_answer(gemini_model, context, question)

        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)
        # st.chat_message(f"Article: {"\n".join([ el[1] for el in retrieved_lines_with_distances])}")
        # st.chat_message(f"Image: {"\n".join([ el[2] for el in retrieved_lines_with_distances])}")


# Display the retrieved lines in a more readable format
st.sidebar.subheader("Retrieved Lines with Distances:")
for idx, (distance, article_url, image_url, text) in enumerate(retrieved_lines_with_distances, 1):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"*Article: {article_url}*")
    st.sidebar.markdown(f"*Image: {image_url}*")
    st.sidebar.markdown(f"*Distance: {distance:.2f}*")
    st.sidebar.markdown(f"**Result {idx}:**")
    st.sidebar.markdown(f"> {text}")