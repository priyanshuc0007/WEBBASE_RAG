import streamlit as st
from rag.loader import load_text
from rag.splitter import split_text
from rag.embedder import embed_text
from rag.store_embedding import store_embeddings
from rag.load_index import load_index
from rag.search_query import search_query
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


def main():
    st.title("RAG Chatbot with Groq API")

    file_path = st.text_input("Enter the path to your text file:", "data/all_text_data.txt")
    query = st.text_input("Enter your query:")

    if st.button("Run"):
        # Load and process text
        text = load_text(file_path)
        chunks = split_text(text, chunk_size=100)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedded_chunks = embed_text(chunks, embeddings)

        store_embeddings(embedded_chunks, embeddings, index_path='data/faiss_index')
        faiss_index = load_index('data/faiss_index', embeddings)

        results = search_query(faiss_index, query)

        for result in results:
            st.write(result)

        # Integrate Groq API for chatbot response
        response = ChatGroq.get_answer(query)
        st.write("Chatbot Response:", response)

if __name__ == '__main__':
    main()
