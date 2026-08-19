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

# Default document length normalization parameter for Okapi BM25 ranking.
# Controls the degree to which document length penalizes term occurrences (0.0 = no penalty, 1.0 = full length scaling).
BM25_B = 0.75

# Cache File Paths
CACHE_DIR = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


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
    Core data structure managing term-to-document mappings, metadata lookups, term frequencies, and document lengths.
    
    Attributes:
        index: Map of stemmed token strings to sets of matching document integer IDs.
        docmap: Map of document integer IDs to complete movie dictionaries.
        term_frequencies: Map of document integer IDs to Counter objects tracking token counts.
        doc_lengths: Map of document integer IDs to total token count (length) of each document.
    """

    def __init__(self) -> None:
        # Initialize empty dictionary mapping tokens (str) -> set of doc IDs (set[int])
        self.index: dict[str, set[int]] = {}
        # Initialize empty dictionary mapping doc ID (int) -> full movie dict (dict)
        self.docmap: dict[int, dict] = {}
        # Initialize empty dictionary mapping doc ID (int) -> token Counter (Counter)
        self.term_frequencies: dict[int, Counter] = {}
        # Initialize dictionary mapping doc ID (int) -> total token count (int)
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Private helper to tokenize document text and add its ID to the inverted index, term frequencies, and document lengths.
        
        Why it exists: Populates internal inverted index, term frequency, and document length state.
        How it works: Tokenizes input text, records token count in doc_lengths, updates term_frequencies Counter, and inserts doc_id into self.index.
        """
        # Tokenize incoming document text (title + description)
        tokens = tokenize_text(text)

        # Count token occurrences within document and assign Counter to doc_id
        self.term_frequencies[doc_id] = Counter(tokens)
        # Store total token count for document length normalization calculations
        self.doc_lengths[doc_id] = len(tokens)

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

    def get_bm25_tf(self, doc_id: int, term: str, b: float = BM25_B, k1: float = BM25_K1) -> float:
        """
        Calculates length-normalized, saturated BM25 Term Frequency (TF) score for a document and term.

        Why it exists: Prevents high raw term counts from disproportionately biasing document relevance ranks
                       while adjusting scores to penalize long documents and reward concise ones.
        How it works: Retrieves raw TF, doc length, and corpus average doc length; computes length normalization factor L;
                       and applies the BM25 TF formula: (tf * (k1 + 1)) / (tf + k1 * L).
        """
        # Retrieve raw integer term occurrence count within the targeted document
        tf = self.get_tf(doc_id, term)
        # Retrieve total token count for document (defaulting to 0 if missing)
        doc_len = self.doc_lengths.get(doc_id, 0)
        # Retrieve average document length across entire corpus
        avgdl = self.__get_avg_doc_length()

        # Handle potential zero average document length to prevent division by zero
        if avgdl == 0:
            length_norm = 1.0
        else:
            # Compute document length normalization factor L = 1 - b + b * (|D| / avgdl)
            length_norm = 1 - b + b * (doc_len / avgdl)

        # Compute length-normalized, saturated BM25 term frequency score
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        """
        Calculates the average document length (avgdl) across all indexed documents.

        Why it exists: Supplies the baseline document length metric required for BM25 length normalization.
        How it works: Sums token counts across self.doc_lengths and divides by total document count in self.docmap.
        """
        # Determine total number of documents in the index
        total_doc_count = len(self.docmap)
        # Return 0.0 float directly if corpus contains no documents
        if total_doc_count == 0:
            return 0.0

        # Sum total token counts across all indexed documents
        summ_doc_length = sum(self.doc_lengths.values())
        # Divide total tokens by document count to compute average document length
        avgdl = summ_doc_length / total_doc_count
        return avgdl

    def bm25(self, doc_id: int, term: str) -> float:
        """
        Calculates the combined Okapi BM25 score for a document and a single term.
        
        Why it exists: Combines non-linear term frequency saturation/length normalization with term rarity weighting.
        How it works: Multiplies length-normalized BM25 TF by BM25 IDF score.
        """
        # Calculate length-normalized BM25 term frequency component
        bm25_tf = self.get_bm25_tf(doc_id, term)
        # Calculate BM25 inverse document frequency component
        bm25_idf = self.get_bm25_idf(term)
        # Combine components via scalar multiplication
        full_bm25 = bm25_tf * bm25_idf
        return full_bm25

    def bm25_search(self, query: str, limit: int = 5) -> List[tuple[int, float]]:
        """
        Executes a full Okapi BM25 search query across the document corpus.

        Why it exists: Ranks corpus documents by cumulative BM25 relevance across all query terms.
        How it works: Tokenizes query, iterates docmap to accumulate term scores, sorts descending, and slices top results.
        """
        # Tokenize and stem user search query into normalized terms
        tokens = tokenize_text(query)
        scores = {}

        # Iterate through every document ID in the indexed corpus
        for doc_id in self.docmap:
            total_score = 0.0
            # Accumulate BM25 score for each query token
            for term in tokens:
                total_score += self.bm25(doc_id, term)

            # Store non-zero relevance scores in accumulator dictionary
            if total_score > 0:
                scores[doc_id] = total_score

        # Sort document scores in descending order (highest relevance first)
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Slice and return top 'limit' (doc_id, score) pairs
        return sorted_results[:limit]

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
        How it works: Ensures cache directory exists, then binary dumps index, docmap, term_frequencies, and doc_lengths.
        """
        # Create 'cache' directory if it doesn't already exist on disk
        os.makedirs("cache", exist_ok=True)
        # Open binary write file handle for index serialization
        with open(INDEX_PATH, "wb") as f:
            # Pickle dump self.index state to binary file
            pickle.dump(self.index, f)
        # Open binary write file handle for docmap serialization
        with open(DOCMAP_PATH, "wb") as f:
            # Pickle dump self.docmap state to binary file
            pickle.dump(self.docmap, f)
        # Open binary write file handle for term frequencies serialization
        with open(TF_PATH, "wb") as f:
            # Pickle dump self.term_frequencies state to binary file
            pickle.dump(self.term_frequencies, f)
        # Open binary write file handle for doc lengths serialization
        with open(DOC_LENGTHS_PATH, "wb") as f:
            # Pickle dump self.doc_lengths state to binary file
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """
        Deserializes index structures from disk via pickle.
        
        Why it exists: Reconstitutes in-memory index structures rapidly during query execution,
                       with fallback mechanisms to rebuild or reconstruct missing metadata dynamically.
        How it works: Opens cached .pkl files. If cache files are missing or incomplete, triggers index rebuilding.
        """
        try:
            # Open binary read file handle for cached index
            with open(INDEX_PATH, "rb") as f:
                # Unpickle binary data and restore to self.index attribute
                self.index = pickle.load(f)
            # Open binary read file handle for cached docmap
            with open(DOCMAP_PATH, "rb") as f:
                # Unpickle binary data and restore to self.docmap attribute
                self.docmap = pickle.load(f)
            # Open binary read file handle for cached term frequencies
            with open(TF_PATH, "rb") as f:
                # Unpickle binary data and restore to self.term_frequencies attribute
                self.term_frequencies = pickle.load(f)
            # Open binary read file handle for cached doc lengths
            with open(DOC_LENGTHS_PATH, "rb") as f:
                # Unpickle binary data and restore to self.doc_lengths attribute
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            # Rebuild and save index if cache files do not exist or are incomplete
            self.build()
            self.save()


def build_command() -> None:
    """
    Handler function for CLI 'build' subcommand.
    
    Why it exists: Orchestrates the build workflow.
    How it works: Instantiates InvertedIndex, builds structures in memory, and persists them to cache.
    """
    # Instantiate new InvertedIndex instance
    idx = InvertedIndex()
    # Populate inverted index, docmap, term frequencies, and document lengths from movies.json
    idx.build()
    # Persist built index data structures to disk cache
    idx.save()        


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """
    Orchestrates execution for the BM25 Term Frequency CLI workflow.

    Why it exists: Decouples business/index logic from raw CLI argument handling.
    How it works: Hydrates index from disk, normalizes input term, computes length-normalized BM25 TF, and returns float score.
    """
    # Instantiate inverted index container
    idx = InvertedIndex()
    # Hydrate index state from binary disk cache (automatically rebuilds if cache files are missing)
    idx.load()

    # Tokenize and stem the single keyword term input
    stemmed_term = tokenize_single_term(term)

    # Compute and return length-normalized saturated BM25 term frequency score
    return idx.get_bm25_tf(doc_id, stemmed_term, b=b, k1=k1)


def main() -> None:
    """
    CLI entry point and command router.
    
    Why it exists: Parses incoming terminal arguments and routes execution to search, build, tf, idf, tfidf, or BM25 workflows.
    How it works: Uses argparse to capture commands/arguments, then evaluates via match/case.
    """
    # Initialize argument parser with CLI description header
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    # Add subparser handler to manage subcommands ('build', 'search', 'tf', 'idf', 'tfidf', 'bm25idf', 'bm25tf', 'bm25search')
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

    # Register 'bm25tf' subcommand with positional doc_id, term, and optional k1 and b arguments
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    # Register 'bm25search' subcommand with query argument and optional limit flag
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit the query results (default: 5)")

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
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = InvertedIndex()   
            try:
                # Hydrate index state from disk cache
                idx.load()
            except FileNotFoundError:
                # Handle missing index files gracefully if build step was skipped
                print("Index not found. Please run build first.")
                return

            # Fetch top (doc_id, score) pairs matching the BM25 search query
            results = idx.bm25_search(args.query, args.limit)

            # Iterate through results and print formatted rank, ID, title, and BM25 score
            for rank, (doc_id, score) in enumerate(results, start=1):
                movie = idx.docmap[doc_id]
                print(f"{rank}. ({doc_id}) {movie['title']} - Score: {score:.2f}")

        case _:
            # Display CLI help menu if command is missing or unrecognized
            parser.print_help()


# Boilerplate execution guard ensuring main() runs only when script is invoked directly
if __name__ == "__main__":
    main()