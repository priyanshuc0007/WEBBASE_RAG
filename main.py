from rag.loader import load_text
from rag.splitter import split_text
from rag.embedder import embed_text
from rag.vector_store import create_faiss_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
import os
from rag.llm import model_load
from rag.prompt_template import get_prompt_template
from langchain.chains import RetrievalQA

def main(file_path: str, index_path='data/faiss_index', query=None):
   
    text = load_text(file_path)

   
    chunks = split_text(text, chunk_size=100)

    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    
    embedded_chunks = embed_text(embeddings, chunks)

    
    vectorstore = create_faiss_vectorstore(chunks, embeddings)
    vectorstore.save_local(faiss_index)
    
    retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
    print(retriever)

    
    relevant_documents = vectorstore.similarity_search(query)
    
    
    print(relevant_documents)

    llm=model_load()
    if llm is None:
        print("Model loading failed. Exiting...")
        return
    prompt=get_prompt_template()

    retrievalQA=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)
    # Call the QA chain with our query.#
    result = retrievalQA.invoke({"query": query})
    print(result['result'])


if __name__ == '__main__':
    file_path = r'E:\RAG_Chatbot_Project\data\all_text_data.txt'  # Replace with your actual file path
    query = 'relinns works on?'  # Replace with your query
    faiss_index=r"E:\WEBBASE_RAG\data"
    main(file_path, query=query)
