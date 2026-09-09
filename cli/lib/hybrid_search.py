# lib/hybrid_search.py

import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def hybrid_score (bm25_score:float, semantic_score: float, alpha: float = 0.5) -> float:
    """
    Calculates weighted combination score balancing keyword and semantic scores.
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score

def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Normalizes a list of numerical scores to a [0.0, 1.0] range using Min-Max scaling.
    Handles empty lists and edge cases where min and max scores are identical.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    # Edge case: avoid division by zero if all scores are identical
    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


class HybridSearch:
    """
    Combines keyword BM25 retrieval and vector semantic search into a hybrid pipeline.
    """

    def __init__(self, documents: list[dict]) -> None:
        """
        Initializes vector semantic search engine and inverted keyword index.
        """
        self.documents = documents

        # Instantiate and load chunked semantic vector search engine
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Instantiate inverted index and ensure cache exists on disk
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """
        Internal helper: executes BM25 search and returns tuples of (doc_id, score).
        """
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        """
        Executes weighted score combination blending normalized BM25 and vector scores.
        """

        fetch_limit = limit * 500

        # 1. Fetch raw search results
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        # Map document objects by ID
        doc_map = {doc["id"]: doc for doc in self.documents}

        # 2. Extract and normalize BM25 scores
        bm25_doc_ids = [doc_id for doc_id, _ in bm25_results]
        raw_bm25_scores = [score for _, score in bm25_results]
        norm_bm25_scores = min_max_normalize(raw_bm25_scores)
        bm25_score_map = dict(zip(bm25_doc_ids, norm_bm25_scores))

        # 3. Extract and normalize semantic scores
        sem_doc_ids = [res["id"] for res in semantic_results]
        raw_sem_scores = [res["score"] for res in semantic_results]
        norm_sem_scores = min_max_normalize(raw_sem_scores)
        sem_score_map = dict(zip(sem_doc_ids, norm_sem_scores))

        # 4. Combine all document IDs into unified candidate map
        all_doc_ids = set(bm25_doc_ids).union(set(sem_doc_ids))
        combined_results = []

        for doc_id in all_doc_ids:
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            bm25_score = bm25_score_map.get(doc_id, 0.0)
            semantic_score = sem_score_map.get(doc_id, 0.0)

            final_score = hybrid_score(bm25_score, semantic_score, alpha)

            combined_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "hybrid_score": final_score,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
            })

        # 5. Sort by hybrid score descending and return top matches
        sorted_results = sorted(
            combined_results, key=lambda x: x["hybrid_score"], reverse=True
        )

        return sorted_results[:limit]


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        """
        Executes Reciprocal Rank Fusion (RRF) combining rank positions from both searchers.
        """
        fetch_limit = limit * 500
        
        # 1. Fetch raw search results
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        # Build a fast lookup dictionary mapping doc_id -> full movie dict
        doc_map = {doc["id"]: doc for doc in self.documents}

        # 2. Map document IDs  to their 1-based rank positions in BM25 results
        bm25_ranks = {}
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            bm25_ranks[doc_id] = rank
        
        # 3. Map document IDs to their 1-based rank positions in Semantic results
        semantic_ranks = {}
        for rank, res in enumerate(semantic_results, start=1):
            semantic_ranks[res["id"]] = rank

        # 4. Get the set of all unique IDs retrieved by either searcher
        all_doc_ids= set(bm25_ranks.keys()).union(set(semantic_ranks.keys()))
        combined_results = []

        # 5. Calculate RRF score for each document across both ranking lists
        for doc_id in all_doc_ids:
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            # Retrieve 1-based ranks (None if the document was not in that result set)
            b_rank = bm25_ranks.get(doc_id)
            s_rank = semantic_ranks.get(doc_id)

            # Calculate individual RRF scores using 1 / (k + rank)
            b_score = rrf_score(b_rank, k) if b_rank is not None else 0.0
            s_score = rrf_score(s_rank, k) if s_rank is not None else 0.0

            # Combined score is the sum of both reciprocal rank scores
            total_rrf_score = b_score + s_score

            combined_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "rrf_score": total_rrf_score,
                "bm25_rank": b_rank,
                "semantic_rank": s_rank,
            })

        # 6. Sort documents by total RRF score in descending order
        sorted_results = sorted(
            combined_results, key=lambda x: x["rrf_score"], reverse=True
        )

        return sorted_results[:limit]