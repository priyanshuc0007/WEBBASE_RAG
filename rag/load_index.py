
from langchain_community.vectorstores import faiss

def load_index(index_path: str) -> faiss:
    """Load a FAISS index."""
    index = faiss.read_index(index_path)
    print("loaded")
    return index