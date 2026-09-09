# lib/keyword_search.py

import json
import math
import os
import pickle
import string
from collections import Counter
from typing import List
from nltk.stem import PorterStemmer

# Default BM25 tuning hyper-parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75   # Document length normalization weight

# Cache directory and storage file paths for index serialization
CACHE_DIR = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


def load_movies() -> List[dict]:
    """
    Loads raw movie records from the JSON dataset.
    """
    with open("data/movies.json") as f:
        data = json.load(f)
    return data["movies"]


def tokenize_single_term(term: str) -> str:
    """
    Tokenizes and stems a single term, validating that exactly one token is produced.
    """
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("Term must be a single term!")
    return tokens[0]


def tokenize_text(text: str) -> List[str]:
    """
    Cleans, strips punctuation/stopwords, and stems an input text string into tokens.
    """
    stemmer = PorterStemmer()
    # Load stopword set from text file and strip punctuation/lowercase
    with open("data/stopwords.txt") as f:
        stopwords = set(
            f.read()
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            .splitlines()
        )
    punc_table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(punc_table).lower()
    # Return list of stemmed words excluding stopwords
    return [stemmer.stem(w) for w in clean_text.split() if w not in stopwords]


class InvertedIndex:
    """
    Inverted index for fast keyword lookups, term statistics, TF-IDF, and BM25 scoring.
    """

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}          # Maps stemmed terms to document IDs
        self.docmap: dict[int, dict] = {}             # Maps document IDs to full document dicts
        self.term_frequencies: dict[int, Counter] = {} # Maps document IDs to term frequency Counters
        self.doc_lengths: dict[int, int] = {}          # Maps document IDs to token count lengths
        self.index_path = INDEX_PATH

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes document text and updates internal index mapping, counters, and lengths.
        """
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        """
        Returns a sorted list of document IDs containing the given stemmed term.
        """
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the raw term frequency (count) for a specific document ID and stemmed term.
        """
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        return doc_counter[term]

    def get_idf(self, term: str) -> float:
        """
        Calculates standard Inverse Document Frequency (IDF) score for a stemmed term.
        """
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        """
        Calculates standard TF-IDF score for a document ID and stemmed term.
        """
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculates the probabilistic BM25 IDF variant for a stemmed term.
        """
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, b: float = BM25_B, k1: float = BM25_K1) -> float:
        """
        Calculates the length-normalized BM25 Term Frequency component for a term in a document.
        """
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avgdl = self.__get_avg_doc_length()
        # Apply length normalization factor based on average document length ratio
        length_norm = 1.0 if avgdl == 0 else (1 - b + b * (doc_len / avgdl))
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        """
        Calculates average document length (token count) across all indexed documents.
        """
        total_doc_count = len(self.docmap)
        if total_doc_count == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / total_doc_count

    def bm25(self, doc_id: int, term: str) -> float:
        """
        Calculates complete BM25 score (BM25 TF * BM25 IDF) for a document ID and term.
        """
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> List[tuple[int, float]]:
        """
         Tokenizes query text, computes total BM25 score per document, and returns top results.
        """
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            total_score = sum(self.bm25(doc_id, term) for term in tokens)
            if total_score > 0:
                scores[doc_id] = total_score
        # Return sorted list of (doc_id, score) tuples up to limit
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def build(self) -> None:
        """
        Builds inverted index and metadata tables from scratch using movie data.
        """
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        """
        Serializes all index structures and metadata tables to pickle cache files.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """
        Deserializes index structures from cache files, or builds them if missing.
        """
        try:
            with open(INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
            with open(TF_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(DOC_LENGTHS_PATH, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            self.build()
            self.save()


# Top-level functions used by CLI handlers
def build_command() -> None:
    """
    CLI command wrapper to instantiate, build, and save the inverted index.
    """
    idx = InvertedIndex()
    idx.build()
    idx.save()


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """
    CLI command wrapper to compute BM25 TF score for a given doc ID and single term.
    """
    idx = InvertedIndex()
    idx.load()
    stemmed_term = tokenize_single_term(term)
    return idx.get_bm25_tf(doc_id, stemmed_term, b=b, k1=k1)