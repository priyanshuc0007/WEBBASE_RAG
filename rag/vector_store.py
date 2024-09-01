from langchain_community.vectorstores import FAISS


def store_embeddings(embedded_chunks, embeddings, index_path='data/faiss_index'):
    faiss_index = FAISS.from_embeddings(embedded_chunks, embeddings)
    faiss_index.save_local(index_path)

def load_index(index_path, embeddings):
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index

def search_query(faiss_index, query, k=5):
    return faiss_index.similarity_search(query, k)
