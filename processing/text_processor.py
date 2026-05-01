import os
import re
import nltk
from nltk.tokenize import sent_tokenize


# Ensure tokenizer is available (runs only if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    # remove extra whitespace and normalize text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk_text(text, chunk_size=3, overlap=1):
    """
    Sentence-based chunking with overlap for better context retention
    """
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def process_file(file_path):
    if file_path.endswith(".txt"):
        text = read_txt(file_path)
    else:
        raise ValueError("Unsupported file type")

    clean = clean_text(text)
    chunks = chunk_text(clean)

    return chunks


if __name__ == "__main__":
    file_path = "sample.txt"

    if not os.path.exists(file_path):
        print("Test file not found!")
    else:
        chunks = process_file(file_path)

        print(f"Total Chunks: {len(chunks)}\n")

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk}\n")