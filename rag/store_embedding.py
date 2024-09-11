
import numpy as np
from langchain_community.vectorstores import FAISS

def store_embeddings(embedded_chunks: list[np.ndarray], index_path: str) -> None:
    """Store embeddings in a FAISS index."""
    index = FAISS.IndexFlatL2(len(embedded_chunks[0]))
    index.add(np.array(embedded_chunks))
    FAISS.write_index(index, index_path)
    print("embedded store")