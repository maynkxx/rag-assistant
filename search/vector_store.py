import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, embeddings, texts):
        if not embeddings:
            return
        self.vectors.extend(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):
        if not self.vectors:
            return []
            
        vectors = np.array(self.vectors)
        query_vec = np.array(query_embedding)

        # Ensure query_vec is 1D
        if query_vec.ndim > 1:
            query_vec = query_vec.flatten()

        # Compute cosine similarity
        # (A . B) / (||A|| * ||B||)
        dot_product = np.dot(vectors, query_vec)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
        
        # Avoid division by zero
        similarities = dot_product / (norms + 1e-10)

        # Get top K results
        top_k = min(top_k, len(self.texts))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.texts[i] for i in top_indices]