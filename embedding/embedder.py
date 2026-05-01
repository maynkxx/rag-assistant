from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer once
vectorizer = TfidfVectorizer()


def get_embeddings(text_chunks):
    """
    Convert text chunks into vector embeddings using TF-IDF
    """
    embeddings = vectorizer.fit_transform(text_chunks).toarray()
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "Refund policy allows returns within 7 days.",
        "Employees get 20 paid leaves per year."
    ]

    embeddings = get_embeddings(sample_chunks)

    print("Embedding shape:", embeddings.shape)