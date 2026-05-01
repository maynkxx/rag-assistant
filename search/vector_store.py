import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, embeddings, texts):
        self.vectors.extend(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):
        vectors = np.array(self.vectors)

        # cosine similarity
        similarities = np.dot(vectors, query_embedding) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.texts[i] for i in top_indices]