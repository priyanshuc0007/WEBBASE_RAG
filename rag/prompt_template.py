
from langchain_core.prompts import PromptTemplate


def get_prompt_template() -> PromptTemplate:
    template = """
    You are an AI assistant tasked with answering questions based on the provided context. Use the information from the context to provide accurate and concise answers. If the context does not provide enough information, state "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=template)    