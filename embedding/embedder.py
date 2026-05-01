from sentence_transformers import SentenceTransformer

# Initialize model once (lightweight and fast)
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(text_chunks):
    """
    Convert text chunks into vector embeddings using sentence-transformers
    """
    if not text_chunks:
        return []
    
    embeddings = model.encode(text_chunks)
    return embeddings.tolist()



if __name__ == "__main__":
    sample_chunks = [
        "Refund policy allows returns within 7 days.",
        "Employees get 20 paid leaves per year."
    ]

    embeddings = get_embeddings(sample_chunks)

    print("Embedding shape:", embeddings.shape)