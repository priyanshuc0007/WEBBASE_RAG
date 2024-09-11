
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 100) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10)
    chunks = splitter.split_text(text)
    print("chunked")
    return chunks