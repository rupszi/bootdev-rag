from sentence_transformers import SentenceTransformer

# MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticSearch:

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_model():
    search_engine = SemanticSearch()
    print(f"Model loaded: {search_engine.model}")
    print(f"Max sequence length: {search_engine.model.max_seq_length}")
    

