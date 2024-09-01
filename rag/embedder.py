
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

def embed_text(chunks: list[str], embeddings: HuggingFaceEmbeddings) -> list[np.ndarray]:
    """Embed text chunks using the Hugging Face model."""
    embedded_chunks = []
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk)
        embedded_chunks.append(embedding)
    return embedded_chunks
