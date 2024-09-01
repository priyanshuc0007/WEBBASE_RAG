
import faiss

def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index."""
    index = faiss.read_index(index_path)
    return index
