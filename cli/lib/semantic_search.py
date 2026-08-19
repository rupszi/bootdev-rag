# Import SentenceTransformer from the sentence_transformers library
# This loads pre-trained transformer models designed to output dense vector embeddings
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticSearch:
    """
    Core semantic search engine class using pre-trained vector embeddings.
    """

    def __init__(self) -> None:
        # Load the 'all-MiniLM-L6-v2' model into memory.
        # This model produces 384-dimensional dense vectors and runs locally.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        """
        Generates a 384-dimensional vector embedding for a given text string.
        """
        # Strip whitespace to guard against empty strings or strings containing only spaces
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")

        # self.model.encode expects a list of text strings.
        # It returns a 2D NumPy array of shape (batch_size, embedding_dim).
        embedding = self.model.encode([text])

        # Return index 0 to retrieve the 1D NumPy array for the single input text
        return embedding[0]


def verify_model() -> None:
    """
    Instantiates SemanticSearch and prints model specifications for verification.
    """
    # Instantiate the search class to trigger model initialization
    search_engine = SemanticSearch()

    # Print the full model architecture summary string
    print(f"Model loaded: {search_engine.model}")

    # Print the maximum token length the model can process per sequence (256)
    print(f"Max sequence length: {search_engine.model.max_seq_length}")


def embed_text(text: str) -> None:
    """
    Generates an embedding for the input text and prints its dimensions.
    """
    # Create an instance of the semantic search module
    srch = SemanticSearch()

    # Generate the 1D NumPy array embedding for the provided text
    embedding = srch.generate_embedding(text)

    # Print original text input
    print(f"Text: {text}")

    # Print the first 3 vector values (floats) to verify scalar range
    print(f"First 3 dimensions: {embedding[:3]}")

    # Print total dimensionality using the .shape property of the NumPy array (384)
    print(f"Dimensions: {embedding.shape[0]}")