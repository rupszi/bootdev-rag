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

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Calculates similarity scores between a search query and all loaded documents,
        returning the top 'limit' closest semantic matches.
        """
        # Guard clause: ensure vector cache and document texts exist before searching
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        # Convert user's raw string search query into a 384-dimensional vector
        embedding = self.generate_embedding(query)

        # Initialize accumulator list to store scored result dictionaries
        result_list = []

        # Iterate through every document and its matching row vector in self.embeddings
        for i in range(len(self.documents)):
            doc_dict = self.documents[i]
            doc_vect = self.embeddings[i]

            # Calculate mathematical similarity between query vector and movie vector
            score = cosine_similarity(embedding, doc_vect)

            # Construct result item containing similarity score and movie details
            result_item = {
                "score": float(score),
                "title": doc_dict["title"],
                "description": doc_dict["description"],
            }

            result_list.append(result_item)

        # Sort all results by 'score' in descending order (highest score first)
        sorted_results = sorted(
            result_list, key=lambda x: x["score"], reverse=True
        )

        # Return only the top N results based on the requested limit
        return sorted_results[:limit]




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
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    """
    Converts a search query into a vector embedding and prints debug metadata.
    """
    # Instantiate search engine helper to access the loaded AI model
    srch = SemanticSearch()

    # Generate vector (384 floats) for the user's raw query string
    embedding = srch.generate_embedding(query)

    # Print input query alongside vector inspection details
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity (angle) between two 1D numerical vectors.
    """
    # Calculate dot product (sum of element-wise multiplication)
    dot_product = np.dot(vec1, vec2)

    # Calculate Euclidean lengths (magnitudes) of both vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Prevent division by zero if either vector is completely empty/zeroed out
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Cosine formula: dot_product divided by product of magnitudes
    return float(dot_product / (norm1 * norm2))


def search_cli(query: str, limit: int = 5) -> None:
    """
    CLI DRIVER: Loads JSON dataset, performs semantic search, and prints formatted results.
    """
    # Instantiate search engine helper
    srch = SemanticSearch()

    # Load raw JSON movie dataset from disk
    with open("data/movies.json", "r") as f:
        movies_data = json.load(f)

    # Load pre-computed vector grid or build it if missing
    srch.load_or_create_embeddings(movies_data["movies"])

    # Execute search algorithm to obtain top matches
    results = srch.search(query, limit)

    # Loop over search results and format standard output for CLI display
    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"  {res['description'][:100]}...\n")


def chunk(text: str, chunk_size: int = 200) -> None:
    """
    Splits text into words on whitespace and groups them into chunks of size `chunk_size`.
    Prints total character count and each chunk with 1-based indexing.
    """
    # Split text on whitespace to get all individual words
    words = text.split()

    chunks = []
    # Step through words list in steps of chunk_size
    for i in range(0, len(words), chunk_size):
        # Slice word list and re-join with single spaces
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    # Print total character length of the original input string
    print(f"Chunking {len(text)} characters")

    # Print numbered chunks starting at index 1
    for i, chunk_str in enumerate(chunks, start=1):
        print(f"{i}. {chunk_str}")