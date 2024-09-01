
import numpy as np
import faiss

def store_embeddings(embedded_chunks: list[np.ndarray], index_path: str) -> None:
    """Store embeddings in a FAISS index."""
    index = faiss.IndexFlatL2(len(embedded_chunks[0]))
    index.add(np.array(embedded_chunks))
    faiss.write_index(index, index_path)