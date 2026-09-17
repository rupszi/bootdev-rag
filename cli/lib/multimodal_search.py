# lib/multimodal_search.py

from typing import cast
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from sentence_transformers import SentenceTransformer
from lib.keyword_search import load_movies


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


class MultimodalSearch:
    def __init__(
        self, documents: list[dict] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        """
        Initializes MultimodalSearch with a CLIP model and optional documents dataset.
        """
        # Load CLIP sentence transformer model
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.documents: list[dict] = documents or []

        # Concatenate title and description for each movie document
        self.texts: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in self.documents
        ]

        # Pre-compute embeddings for all text documents if dataset is provided
        if self.texts:
            self.text_embeddings = self.model.encode(
                self.texts, show_progress_bar=True
            )
        else:
            self.text_embeddings = []

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from image_path and generates its vector embedding.
        """
        raw_img = Image.open(image_path)
        image = cast(PILImage, raw_img)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        """
        Generates an image embedding and computes cosine similarity against all text embeddings.
        Returns top_k matching documents with their similarity scores.
        """
        # Generate embedding for query image
        image_emb = self.embed_image(image_path)

        results = []
        # Calculate cosine similarity against each document's text embedding
        for idx, doc in enumerate(self.documents):
            text_emb = self.text_embeddings[idx]
            sim = cosine_similarity(image_emb, text_emb)
            results.append(
                {
                    "id": doc.get("id"),
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": sim,
                }
            )

        # Sort results in descending order by similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]


def verify_image_embedding(image_path: str) -> None:
    """
    Instantiates MultimodalSearch without documents and prints the image embedding shape.
    """
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    """
    Loads movie dataset, creates MultimodalSearch instance, and returns image search results.
    """
    movies = load_movies()
    searcher = MultimodalSearch(documents=movies)
    return searcher.search_with_image(image_path, top_k=5)