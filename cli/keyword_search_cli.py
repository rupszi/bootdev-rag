# Import argparse to handle Command Line Interface (CLI) arguments and subcommands
import argparse
# Import json to parse raw movie data stored in JSON format
import json
# Import math for logarithmic calculations used in Inverse Document Frequency (IDF)
import math
# Import os to check and create cache directories on disk
import os
# Import pickle to serialize/deserialize Python dictionary state to binary disk files
import pickle
# Import string for built-in collections of ASCII punctuation characters
import string
# Import Counter to count term occurrences per document efficiently
from collections import Counter
# Import type hints to enforce clearer function signatures and contract boundaries
from typing import List
# Import PorterStemmer to reduce words to their stem/root form (e.g., 'running' -> 'run')
from nltk.stem import PorterStemmer

"""
Constants
"""
# Default term frequency saturation parameter for Okapi BM25 ranking.
# Controls how quickly additional term occurrences diminish in marginal relevance (typically between 1.2 and 2.0).
BM25_K1 = 1.5


def load_movies() -> List[dict]:
    """
    Reads movie dataset from disk.
    
    Why it exists: Serves as the raw data source during the build phase.
    How it works: Opens data/movies.json, parses JSON structure, and returns the 'movies' list.
    """
    # Open the raw movies JSON file in read mode using a context manager
    with open("data/movies.json") as f:
        # Load and parse raw JSON into a Python dictionary structure
        data = json.load(f)
    # Extract and return the array of movie objects stored under the 'movies' key
    return data["movies"]


def tokenize_single_term(term: str) -> str:
    """
    Tokenizes and stems a single term input.

    Why it exists: Ensures single keyword CLI inputs (e.g., for TF/IDF) undergo 
                   identical normalization as document text while validating input count.
    How it works: Passes text to tokenize_text and enforces that exactly one token is returned.
    """
    # Tokenize input using standard preprocessing pipeline
    tokens = tokenize_text(term)
    # Ensure input resolves to exactly one token
    if len(tokens) != 1:
        raise ValueError("Term must be a single term!")
    return tokens[0]


def tokenize_text(text: str) -> List[str]:
    """
    Normalizes and splits raw text into searchable stemmed tokens.
    
    Why it exists: Standardizes both indexed documents and search queries so term matches succeed.
    How it works: Removes punctuation, lowercases, drops stopwords, and stems remaining terms.
    """
    # Instantiate the PorterStemmer for algorithmic word suffix stripping
    stemmer = PorterStemmer()

    # Open stopwords file to load noise words (e.g., 'the', 'is', 'and')
    with open("data/stopwords.txt") as f:
        # Load stopwords, remove punctuation, normalize to lowercase, and store as a set for O(1) lookup
        stopwords = set(
            f.read()
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            .splitlines()
        )

    # Construct a translation table mapping all punctuation characters to None
    punc_table = str.maketrans("", "", string.punctuation)
    
    # Apply punctuation removal and convert input string to lowercase
    clean_text = text.translate(punc_table).lower()

    # Split cleaned text into words, filter out stopwords, stem each word, and return the token list
    return [stemmer.stem(w) for w in clean_text.split() if w not in stopwords]


class InvertedIndex:
    """
    Core data structure managing term-to-document mappings, metadata lookups, and term frequencies.
    
    Attributes:
        index: Map of stemmed token strings to sets of matching document integer IDs.
        docmap: Map of document integer IDs to complete movie dictionaries.
        term_frequencies: Map of document integer IDs to Counter objects tracking token counts.
    """

    def __init__(self) -> None:
        # Initialize empty dictionary mapping tokens (str) -> set of doc IDs (set[int])
        self.index: dict[str, set[int]] = {}
        # Initialize empty dictionary mapping doc ID (int) -> full movie dict (dict)
        self.docmap: dict[int, dict] = {}
        # Initialize empty dictionary mapping doc ID (int) -> token Counter (Counter)
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Private helper to tokenize document text and add its ID to the inverted index and term frequencies.
        
        Why it exists: Populates internal inverted index and term frequency state.
        How it works: Tokenizes input text, updates term_frequencies Counter, and inserts doc_id into self.index.
        """
        # Tokenize incoming document text (title + description)
        tokens = tokenize_text(text)

        # Count token occurrences within document and assign Counter to doc_id
        self.term_frequencies[doc_id] = Counter(tokens)

        # Iterate over each stemmed token generated from the text
        for token in tokens:
            # Ensure the token key exists in self.index (defaulting to empty set) and add doc_id to it
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        """
        Retrieves sorted document IDs matching a given stemmed term.
        
        Why it exists: Provides fast O(1) lookup of document IDs for search terms.
        How it works: Fetches the token's set from self.index (or empty set if missing) and returns it sorted.
        """
        # Retrieve term's document ID set from index (default empty set if missing) and return as sorted list
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Retrieves term frequency count for a given document and term.

        Why it exists: Enables fast single-term frequency lookup per document.
        How it works: Fetches document Counter from term_frequencies and returns key count (defaulting to 0).
        """
        # Retrieve term counter for doc_id (default to empty Counter) and return count for specified term
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        return doc_counter[term]

    def get_idf(self, term: str) -> float:
        """
        Calculates Inverse Document Frequency (IDF) for a stemmed term.

        Why it exists: Measures term rarity across the entire corpus.
        How it works: Divides total documents by documents containing the term (with smoothing) and takes the natural log.
        """
        # Calculate total document count across the corpus
        total_doc_count = len(self.docmap)

        # Calculate number of documents containing the tokenized term
        term_match_doc_count = len(self.get_documents(term))

        # Compute smoothed logarithmic Inverse Document Frequency (IDF)
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:
        """
        Calculates Term Frequency-Inverse Document Frequency (TF-IDF) score.

        Why it exists: Measures the relative importance of a term within a specific document context.
        How it works: Multiplies Term Frequency (TF) by Inverse Document Frequency (IDF).
        """
        # Calculate Term Frequency for document and term
        tf = self.get_tf(doc_id, term)
        # Calculate Inverse Document Frequency for term
        idf = self.get_idf(term)
        # Combine metrics via multiplication
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculates BM25 Inverse Document Frequency (IDF) score for a stemmed term.

        Why it exists: Applies Okapi BM25's non-linear IDF curve to weight term rarity.
        How it works: Evaluates ln((N - df + 0.5) / (df + 0.5) + 1) using corpus size N and match count df.
        """
        # Calculate total document count across the corpus
        total_doc_count = len(self.docmap)

        # Calculate number of documents containing the tokenized term
        term_match_doc_count = len(self.get_documents(term))

        # Compute BM25 non-linear logarithmic IDF score
        bm25_idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        """
        Calculates saturated BM25 Term Frequency (TF) score for a document and term.

        Why it exists: Prevents high raw term counts from disproportionately biasing document relevance ranks.
        How it works: Retrieves raw term frequency and applies non-linear saturation curve (tf * (k1 + 1)) / (tf + k1).
        """
        # Retrieve raw integer term occurrence count within the targeted document
        tf = self.get_tf(doc_id, term)

        # Calculate saturated term frequency score using parameter k1
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    def build(self) -> None:
        """
        Executes complete index construction from raw movie data.
        
        Why it exists: Precomputes the index structures before saving to disk.
        How it works: Loads movies, maps IDs to objects in docmap, combines text, and indexes tokens.
        """
        # Load raw movie list from disk
        movies = load_movies()
        # Iterate over each movie object in dataset
        for m in movies:
            # Extract document ID from movie record
            doc_id = m["id"]
            # Store complete movie object in docmap under its document ID
            self.docmap[doc_id] = m
            # Concatenate title and description to construct full searchable text corpus
            text = f"{m['title']} {m['description']}"
            # Pass document ID and text corpus to internal indexer
            self.__add_document(doc_id, text)

    def save(self) -> None:
        """
        Serializes index structures to disk via pickle.
        
        Why it exists: Persists index state so future search CLI runs skip parsing raw JSON.
        How it works: Ensures cache directory exists, then binary dumps index, docmap, and term_frequencies.
        """
        # Create 'cache' directory if it doesn't already exist on disk
        os.makedirs("cache", exist_ok=True)
        # Open binary write file handle for index serialization
        with open("cache/index.pkl", "wb") as f:
            # Pickle dump self.index state to binary file
            pickle.dump(self.index, f)
        # Open binary write file handle for docmap serialization
        with open("cache/docmap.pkl", "wb") as f:
            # Pickle dump self.docmap state to binary file
            pickle.dump(self.docmap, f)
        # Open binary write file handle for term frequencies serialization
        with open("cache/term_frequencies.pkl", "wb") as f:
            # Pickle dump self.term_frequencies state to binary file
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        """
        Deserializes index structures from disk via pickle.
        
        Why it exists: Reconstitutes in-memory index structures rapidly during query execution.
        How it works: Opens cached .pkl files and unpickles data back into index, docmap, and term_frequencies.
        """
        # Open binary read file handle for cached index
        with open("cache/index.pkl", "rb") as f:
            # Unpickle binary data and restore to self.index attribute
            self.index = pickle.load(f)
        # Open binary read file handle for cached docmap
        with open("cache/docmap.pkl", "rb") as f:
            # Unpickle binary data and restore to self.docmap attribute
            self.docmap = pickle.load(f)
        # Open binary read file handle for cached term frequencies
        with open("cache/term_frequencies.pkl", "rb") as f:
            # Unpickle binary data and restore to self.term_frequencies attribute
            self.term_frequencies = pickle.load(f)


def build_command() -> None:
    """
    Handler function for CLI 'build' subcommand.
    
    Why it exists: Orchestrates the build workflow.
    How it works: Instantiates InvertedIndex, builds structures in memory, and persists them to cache.
    """
    # Instantiate new InvertedIndex instance
    idx = InvertedIndex()
    # Populate inverted index, docmap, and term frequencies from movies.json
    idx.build()
    # Persist built index data structures to disk cache
    idx.save()        


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    """
    Orchestrates execution for the BM25 Term Frequency CLI workflow.

    Why it exists: Decouples business/index logic from raw CLI argument handling.
    How it works: Hydrates index from disk, normalizes input term, computes saturated BM25 TF, and returns float score.
    """
    # Instantiate inverted index container
    idx = InvertedIndex()
    try:
        # Hydrate index state from binary disk cache
        idx.load()
    except FileNotFoundError:
        # Print fallback notice if cache does not exist
        print("Index not found. Please run build first.")
        return 0.0

    # Tokenize and stem the single keyword term input
    stemmed_term = tokenize_single_term(term)

    # Compute and return saturated BM25 term frequency score
    return idx.get_bm25_tf(doc_id, stemmed_term, k1)


def main() -> None:
    """
    CLI entry point and command router.
    
    Why it exists: Parses incoming terminal arguments and routes execution to search, build, tf, idf, tfidf, or BM25 workflows.
    How it works: Uses argparse to capture commands/arguments, then evaluates via match/case.
    """
    # Initialize argument parser with CLI description header
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    # Add subparser handler to manage subcommands ('build', 'search', 'tf', 'idf', 'tfidf', 'bm25idf', 'bm25tf')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register 'build' subcommand with help text
    subparsers.add_parser("build", help="Build and save the inverted index")

    # Register 'search' subcommand with help text
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    # Define required positional string argument 'query' for search command
    search_parser.add_argument("query", type=str, help="Search query")

    # Register 'tf' subcommand with help text
    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single term to query")

    # Register 'idf' subcommand with help text
    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Single term to query")

    # Register 'tfidf' subcommand with help text
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the combined term frequency and inverse document frequency for a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single term to query")

    # Register 'bm25idf' subcommand with help text
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # Register 'bm25tf' subcommand with positional doc_id, term, and optional k1 argument
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")

    # Parse command-line arguments passed at execution time
    args = parser.parse_args()

    # Route execution based on command argument passed
    match args.command:
        case "search":
            # Instantiate InvertedIndex container
            idx = InvertedIndex()
            try:
                # Attempt to hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Output initial search query status message
            print(f"Searching for: {args.query}")
            # Initialize empty set to collect matching document IDs without duplicates
            matching_doc_ids = set()
            # Tokenize and stem user's raw search query
            tokenized_query = tokenize_text(args.query)

            # Loop over each token extracted from user query
            for token in tokenized_query:
                # Retrieve document IDs matching current query token
                doc_ids = idx.get_documents(token)
                # Loop through retrieved document IDs
                for doc_id in doc_ids:
                    # Collect document ID into set (automatically deduplicates)
                    matching_doc_ids.add(doc_id)

            # Initialize empty list to hold final resolved movie objects
            result = []
            # Sort document IDs numerically for deterministic result order
            for doc_id in sorted(matching_doc_ids):
                # Retrieve full movie record from docmap using doc_id and append to results
                result.append(idx.docmap[doc_id])
                # Cap result set at top 5 matching movies
                if len(result) >= 5:
                    break

            # Print formatted top 5 results with index rank, title, and movie ID
            for i, movie in enumerate(result, start=1):
                print(f"{i}. {movie['title']}, (ID: {movie['id']})")

        case "build":
            # Delegate execution to build command handler
            build_command()

        case "tf":
            idx = InvertedIndex()
            try:
                # Hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Tokenize and stem the single term input from CLI
            stemmed_term = tokenize_single_term(args.term)

            # Look up term frequency in the given document ID
            tf = idx.get_tf(args.doc_id, stemmed_term)

            # Output integer frequency result to stdout
            print(tf)

        case "idf":
            idx = InvertedIndex()   
            try:
                # Hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Tokenize and stem the single term input from CLI
            stemmed_term = tokenize_single_term(args.term)

            # Compute smoothed logarithmic Inverse Document Frequency (IDF)
            idf = idx.get_idf(stemmed_term)

            # Output formatted IDF value rounded to 2 decimal places
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()   
            try:
                # Hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Tokenize and stem the single term input from CLI
            stemmed_term = tokenize_single_term(args.term)

            # Calculate TF-IDF score for the given document ID and term
            tf_idf = idx.get_tfidf(args.doc_id, stemmed_term)

            # Output formatted TF-IDF value rounded to 2 decimal places
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            idx = InvertedIndex()   
            try:
                # Hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Tokenize and stem the single term input from CLI
            stemmed_term = tokenize_single_term(args.term)
            bm25_idf = idx.get_bm25_idf(stemmed_term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            # Delegate to specialized command handler and print output formatted to 2 decimal places
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            # Display CLI help menu if command is missing or unrecognized
            parser.print_help()


# Boilerplate execution guard ensuring main() runs only when script is invoked directly
if __name__ == "__main__":
    main()