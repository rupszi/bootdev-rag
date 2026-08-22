import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import re


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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None


    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Turns every movie description in our list into semantic chunks,

        generates vector embeddings for all chunks, and saves both embeddings
        and metadata to disk.
        """
        # 1. Store the raw list of movie dictionaries on the instance
        self.documents = documents

        # 2. Build a fast key-value lookup map from movie ID to full dictionary
        self.document_map = {doc["id"]: doc for doc in documents}

        # Flat list to collect chunk text strings across ALL documents for batch encoding
        all_chunks = []

        # List of metadata dicts tracking positional provenance for every chunk
        self.chunk_metadata = []

        # 3. Process every document and generate text chunks
        for movie_idx, doc in enumerate(self.documents):
            # Skip documents with missing or whitespace-only descriptions
            if not doc["description"].strip():
                continue

            # Split description into overlapping sentence chunks (max 4 sentences, 1 sentence overlap)
            chunks = semantic_chunk(doc["description"], 4, 1)

            # Record each individual chunk and its positional metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                self.chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,  # Index of parent movie in self.documents
                        "chunk_idx": chunk_idx,  # Position of chunk within this movie
                        "total_chunks": len(
                            chunks
                        ),  # Total chunk count for this movie
                    }
                )

        # 4. Batch encode all collected chunk text strings into a 2D numpy matrix of vectors
        self.chunk_embeddings = self.model.encode(
            all_chunks, show_progress_bar=True
        )

        # 5. Serialize vector matrix to binary NumPy format on disk (.npy)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        # 6. Write chunk metadata and overall total count to disk as formatted JSON
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        # 7. Return generated vector matrix
        return self.chunk_embeddings



    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Loads pre-computed chunk embeddings and metadata from disk cache if present;

        otherwise builds them from scratch.
        """
        # 1. Store state and fast lookup map
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        # 2. Check if BOTH cache files exist on disk
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            print("Loading chunk embeddings and metadata from cache...")

            # Load the 2D vector matrix directly from .npy file
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")

            # Load metadata JSON and extract the 'chunks' list
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        # 3. Fall back to generating embeddings if cache is missing
        print("Cache missing. Generating chunk embeddings...")
        return self.build_chunk_embeddings(documents)


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


def chunk(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    """
    Splits text into words on whitespace and groups them into chunks of size `chunk_size`.
    Prints total character count and each chunk with 1-based indexing.
    """
    # Guard against invalid overlap configuration
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size.")
    
    # Split text on whitespace to get all individual words
    words = text.split()

    # Guard clause for empty input
    if not words:
        print(f"Chunking {len(text)} characters")

    chunks = []
    i = 0
    stride = chunk_size - overlap

    # 2. Slide window across the word list using a while loop
    while i <len(words):
        # Extract chunk slice from current index
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

        # Stop if this chunk already reached or passed the end of words list
        if i + chunk_size >= len(words):
            break

        # Advance pointer by stride (chunk_size - overlap)
        i += stride

    # Print total character length of the original input string
    print(f"Chunking {len(text)} characters")

    # Print numbered chunks starting at index 1
    for i, chunk_str in enumerate(chunks, start=1):
        print(f"{i}. {chunk_str}")


def semantic_chunk(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    """
    Splits input text into individual sentences using regex lookbehinds and groups them into
    chunks of up to `max_chunk_size` sentences with `overlap` sentence continuity.
    Returns a list of chunk strings and prints formatted CLI output.
    """
    # Guard against invalid overlap configurations where stride would be <= 0
    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be strictly less than max_chunk_size.")

    # Split text into sentences based on punctuation boundary followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Output header matching required test output format
    print(f"Semantically chunking {len(text)} characters")

    # Guard clause for empty/whitespace-only input strings
    if not text or not sentences or sentences == [""]:
        return []

    chunks = []
    i = 0
    stride = max_chunk_size - overlap

    # 2. Slide window across the word list using a while loop
    while i <len(sentences):
        # Extract chunk slice from current index
        chunk_sentences = sentences[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))

        # Stop if this chunk already reached or passed the end of words list
        if i + max_chunk_size >= len(sentences):
            break

        # Advance pointer by stride (chunk_size - overlap)
        i += stride

    # Print numbered chunks starting at 1
    for idx, chunk_str in enumerate(chunks, start=1):
        print(f"{idx}. {chunk_str}")

    # Return list of chunk strings as required by the assignment spec
    return chunks

def embed_chunks(documents: list[dict]):
    """Instantiates the ChunkedSemanticSearch engine, builds or loads cached

    chunk embeddings, and reports the resulting vector count.
    """
    chunk_srch = ChunkedSemanticSearch()
    embeddings = chunk_srch.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")