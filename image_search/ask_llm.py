import google.generativeai as genai



def get_llm_answer(client: genai.GenerativeModel, context: str, question: str):
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    response = client.generate_content(USER_PROMPT)
    return response.text