import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load environment variables from .env file
load_dotenv()

def initialize_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    # Get the token from the environment variables
    token = os.getenv('HF_TOKEN')
    
    if token is None:
        raise ValueError("HF_TOKEN environment variable not set")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        token=token
    )
print("model loaded")


def embed_text(embeddings: HuggingFaceEmbeddings, chunks: list[str]) -> list[np.ndarray]:
    """Embed a list of text chunks using Hugging Face embeddings."""
    embedded_chunks = []
    
    for chunk in chunks:
        # Embed each individual chunk of text
        embedding = embeddings.embed_query(chunk)
        embedded_chunks.append(embedding)
    
    print(f"Successfully embedded {len(chunks)} chunks.")
    return embedded_chunks
print("embed_text")
