import streamlit as st
from rag.loader import load_text
from rag.splitter import split_text
from rag.embedder import embed_text
from rag.vector_store import create_faiss_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from rag.llm import model_load
from rag.prompt_template import get_prompt_template
from langchain.chains import RetrievalQA

# Helper function to process and prepare the vector store
def process_data(file_path, chunk_size, index_path):
    text = load_text(file_path)
    chunks = split_text(text, chunk_size=chunk_size)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    embedded_chunks = embed_text(embeddings, chunks)
    vectorstore = create_faiss_vectorstore(chunks, embeddings)
    vectorstore.save_local(index_path)
    
    return vectorstore

# Function to generate the answer based on query and vector store
def generate_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = model_load()
    prompt = get_prompt_template()

    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = retrievalQA.invoke({"query": query})
    return result['result'], result['source_documents']

# Streamlit App layout and flow
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")

    st.title("RAG Chatbot powered by LLM")
    st.markdown("### Streamline your data and ask the right questions.")
    
    # File upload section
    st.sidebar.header("Upload Text Data")
    file_path = st.sidebar.text_input("Enter the file path:", r'E:\RAG_Chatbot_Project\data\all_text_data.txt')
    chunk_size = st.sidebar.slider("Select Chunk Size:", 50, 500, 100)
    index_path = st.sidebar.text_input("FAISS Index Path:", r"E:\WEBBASE_RAG\data")

    if st.sidebar.button("Process Data"):
        st.session_state.vectorstore = process_data(file_path, chunk_size, index_path)
        st.sidebar.success("Data processed successfully.")
    
    # Check if the vector store is ready
    if 'vectorstore' not in st.session_state:
        st.info("Please process the data first.")
        return

    # Query input section
    st.header("Ask a Question")
    query = st.text_input("Enter your query:", value='relinns works on?')

    if st.button("Get Answer"):
        if query:
            with st.spinner("Retrieving answer..."):
                answer, documents = generate_answer(st.session_state.vectorstore, query)
                st.write(f"**Answer:** {answer}")

            with st.expander("View Relevant Documents"):
                for doc in documents:
                    st.write(doc)

if __name__ == '__main__':
    main()
