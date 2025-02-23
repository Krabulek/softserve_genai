import google.generativeai as genai
from google.generativeai.types import GenerationConfig


config = GenerationConfig(
    temperature=0.2, top_k=32
)


def get_llm_answer(client: genai.GenerativeModel, context: str, question: str):
    USER_PROMPT = f"""Use the following pieces of information enclosed in <context> tags \
to provide an answer to the question enclosed in <question> tags.
Provide an answer which is factually correct and based in the context.
<context>
{context}
</context>
<question>
{question}
</question>`
"""

    response = client.generate_content(USER_PROMPT, generation_config=config)
    return response.text