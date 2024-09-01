
def load_text(file_path: str) -> str:
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
