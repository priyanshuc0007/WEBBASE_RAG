from langchain_community.vectorstores import FAISS
import os

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

def create_faiss_vectorstore(chunks: list[str], embeddings: HuggingFaceEmbeddings):
    """Create a FAISS vectorstore from a list of text chunks and save it."""
    
    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create FAISS vectorstore from the documents and embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

print("vector is store and file is created")

def load_index(index_path, embeddings):
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index

def search_query(faiss_index, query, k=5):
    return faiss_index.similarity_search(query, k)