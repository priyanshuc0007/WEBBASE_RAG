
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss

def search_query(query: str, index: faiss.Index, embeddings: HuggingFaceEmbeddings, text_data: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Search for a query in the index and limit the results."""
    query_embedding = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)  # retrieve top 'top_k' nearest neighbors
    results = [(text_data[idx][:200] + '...', dist) for idx, dist in zip(indices[0], distances[0])]  # truncate results to 200 characters
    return results