import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    """
    Search engine that understands the meaning of words, not just exact keyword matches.
    """

    def __init__(self) -> None:
        # Load our local AI model. Think of this model as a translator that turns
        # English text into a list of 384 numbers representing its overall vibe/meaning.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Set up empty slots to store our vector grid, raw movies, and quick lookup map
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict = {} 

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Converts a single string of text into a list of 384 numbers (vector).
        """
        # Guard against blank or whitespace-only inputs
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")

        # self.model.encode expects a list of strings, e.g. ["star wars"]
        # It returns a 2D array of vectors, so [0] grabs the vector for our single input
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Turns every movie in our list into a vector and saves the result to disk.
        """
        # Save raw movie dictionaries internally
        self.documents = documents

        # Create a fast lookup map: movie_id -> full movie dictionary
        self.document_map = {doc["id"]: doc for doc in documents}

        # Combine title and description into one sentence per movie.
        # Example: "The Dark Knight: When the menace known as the Joker..."
        # This gives the AI model full context to convert into numbers.
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]

        # Convert all movie strings into vectors at once (batching is much faster)
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        # Save the vector grid to disk so we don't have to re-compute it every run
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Loads saved vectors from disk if available; otherwise builds them from scratch.
        """
        # Populate our internal document states
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        # Check if our "save file" exists in the cache folder
        cache_path = "cache/movie_embeddings.npy"
        if os.path.exists(cache_path):
            # Load the cached grid of numbers from disk
            self.embeddings = np.load(cache_path)

            # Make sure the cached vector count matches the current movie count
            if self.embeddings is not None and len(self.embeddings) == len(documents):
                return self.embeddings

        # If cache is missing or out of date, compute vectors from scratch
        return self.build_embeddings(documents)


def verify_model() -> None:
    """
    Prints basic information about the loaded AI model.
    """
    search_engine = SemanticSearch()
    print(f"Model loaded: {search_engine.model}")
    print(f"Max sequence length: {search_engine.model.max_seq_length}")


def embed_text(text: str) -> None:
    """
    Generates a vector for user-provided text and prints its size details.
    """
    srch = SemanticSearch()
    embedding = srch.generate_embedding(text)

    print(f"Text: {text}")
    # Print the first 3 numbers out of 384 just to see what the numbers look like
    print(f"First 3 dimensions: {embedding[:3]}")
    # Print the total count of numbers in the vector (384)
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    """
    Loads movie JSON data, generates/loads vector embeddings, and prints matrix shape.
    """
    srch = SemanticSearch()

    # Open and parse the JSON dataset file containing our movie list
    with open("data/movies.json", "r") as f:
        movies_data = json.load(f)

    # Extract the actual list of movie dictionaries
    documents = movies_data["movies"]

    # Load vectors from cache or calculate new ones
    embeddings = srch.load_or_create_embeddings(documents)

    # Assure type checker that embeddings is valid
    if embeddings is None:
        raise ValueError("Failed to generate or load embeddings.")

    # Output total count of movies and dimensionality (e.g. 5000 vectors in 384 dimensions)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")