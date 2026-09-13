# lib/multimodal_search.py

from typing import cast
from PIL import Image
from PIL.Image import Image as PILImage
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        """
        Initializes the multimodal search engine using a CLIP sentence-transformer model.
        """
        # Explicitly annotate self.model to satisfy Pylance attribute resolution
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        """
        Loads an image from the given path and generates its vector embedding.
        """
        # Open image using PIL
        raw_img = Image.open(image_path)

        # Cast image to PILImage to satisfy Pyright type bounds
        image = cast(PILImage, raw_img)

        # Generate embedding vector
        embeddings = self.model.encode([image])

        return embeddings[0]


def verify_image_embedding(image_path: str) -> None:
    """
    Instantiates MultimodalSearch, generates an embedding for the image, and prints its dimension shape.
    """
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)

    # Print the shape in the required format
    print(f"Embedding shape: {embedding.shape[0]} dimensions")