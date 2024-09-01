
from rag.loader import load_text
from rag.splitter import split_text
from rag.embedder import embed_text
from rag.store_embeding import store_embeddings
from rag.load_index import load_index
from rag.search_query import search_query
from langchain_community.embeddings import HuggingFaceEmbeddings

def main(file_path: str, index_path='data/faiss_index', query=None):
   
    text = load_text(file_path)

    chunks = split_text(text, chunk_size=100)
   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded_chunks = embed_text(chunks, embeddings)
    
    store_embeddings(embedded_chunks, index_path)
   
    index = load_index(index_path)
   
    if query is not None:
        results = search_query(query, index, embeddings, chunks)
        for result in results:
            print(result)

if __name__ == '__main__':
    file_path =  r'E:\RAG_Chatbot_Project\data\all_text_data.txt'  # replace with your file path
    query = 'LET’S WORK TOGETHER'  # replace with your query
    main(file_path, query=query)
