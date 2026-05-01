from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "Refund policy allows returns within 7 days.",
        "Employees get 20 paid leaves per year."
    ]

    embeddings = get_embeddings(sample_chunks)

    print("Embedding shape:", embeddings.shape)