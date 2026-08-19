# Import SentenceTransformer from sentence_transformers package
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    """
    Core semantic search engine class utilizing pre-trained vector embeddings.
    """

    def __init__(self) -> None:
        # Load the pre-trained all-MiniLM-L6-v2 sentence transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_model():
    """
    Instantiates SemanticSearch and prints model specifications for CLI verification.
    """
    # Instantiate the SemanticSearch class (triggers model loading)
    search_engine = SemanticSearch()

    # Print required output specifications
    print(f"Model loaded: {search_engine.model}")
    print(f"Max sequence length: {search_engine.model.max_seq_length}")
    

