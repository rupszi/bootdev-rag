Here is an expanded, production-ready README.md that elevates the project documentation without adding unnecessary fluff or overcomplicating the underlying design.
Fast Keyword Search Engine in Python

A lightweight, zero-dependency Inverted Index Search Engine built from scratch in Python. It processes unstructured document datasets once, constructs an in-memory postings map, and serializes artifacts to disk to enable O(1) term lookups and near-instant retrieval.
Purpose & Problem Statement

When querying thousands of raw JSON files or unindexed records, traditional search implementations default to linear scans:

    Naive Approach (Linear Scan): Iterates over every document sequentially, executing substring or regex checks. Time complexity grows linearly as O(N×L), where N is document count and L is average document length.

    Inverted Index Approach: Maps normalized terms (stems) directly to the set of document IDs containing them. Lookups execute in O(1) average time, shifting indexing complexity to a single offline build phase.

This engine transforms arbitrary document collections (e.g., movie titles and descriptions) into a binary postings list, eliminating runtime I/O and tokenization overhead on user queries.
System Architecture

                       ┌──────────────────────────────────────┐
                       │    RAW DATA INGESTION & PARSING      │
                       │  movies.json  +  stopwords.txt       │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │   TEXT NORMALIZATION & TOKENIZATION  │
                       │ Strip Punctuation -> Lowercase       │
                       │ Drop Stopwords    -> Stem Roots      │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │    BUILD INVERTED INDEX & PERSIST    │
                       │ Term -> Set[Doc IDs] & Doc ID Map    │
                       │ Serialized to cache/*.pkl via Pickle │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │         CLI SEARCH EXECUTION         │
                       │ Load cache/*.pkl -> Normalize Query  │
                       │ O(1) Set Lookup -> Sort IDs -> Print  │
                       └──────────────────────────────────────┘

Architectural Walkthrough
1. Text Normalization Pipeline (tokenize_text)

To ensure robust match recall across grammatical variants, raw text passes through four sequential transformation stages:

Raw Input:  "The brave knight fought bravely!"
1. Case Fold:   "the brave knight fought bravely!"
2. Strip Punc:  "the brave knight fought bravely"
3. Filter Stop: ["brave", "knight", "fought", "bravely"]   (filters terms in stopwords.txt)
4. Stem Roots:  ["brav", "knight", "fought", "brav"]     (PorterStemmer reduction)

Stemming guarantees that queries like "bravery", "brave", or "bravely" normalize to the same root stem ("brav"), matching identical document IDs.
2. Primary Data Structures (InvertedIndex)

The engine maintains two core mapping structures:

    Postings Index (self.index): dict[str, set[int]]

    Maps normalized stems directly to sets of document IDs.
    Python

    {
        "brav": {2054, 2577, 4101, 4104},
        "space": {101, 402, 988},
        "knight": {2054, 8812}
    }

    Document Map (self.docmap): dict[int, dict]

    Maps unique document IDs to their original raw payloads for zero-latency result hydration.
    Python

    {
        2054: {"id": 2054, "title": "Mohawk", "description": "A brave warrior..."},
        4101: {"id": 4101, "title": "Tuck Everlasting", "description": "A story about..."}
    }

3. Disk Serialization & Caching

    build Phase: Tokenizes dataset records, populates the InvertedIndex in memory, and serializes structures to cache/index.pkl and cache/docmap.pkl using standard library pickle.

    search Phase: Bypasses source files entirely by loading the binary artifacts directly into memory in milliseconds.

4. Deterministic Query Execution

    Query Stemming: Normalizes user input ("brave" → ["brav"]).

    Postings Retrieval: Performs O(1) dictionary lookups (idx.get_documents("brav")).

    Deterministic Ordering: Converts unordered Python sets into sorted sequences (sorted(matching_doc_ids)), ensuring stable outputs across distinct runtime environments.

    Hydration: Resolves top document IDs to original metadata titles via self.docmap.

Usage Guide & Operations
Installation & Environment Setup

Ensure uv is installed, then sync project dependencies:
Bash

uv sync

Build Index

Parse raw input datasets and generate binary cache files in cache/:
Bash

uv run cli/keyword_search_cli.py build

Run Queries

Execute term lookups against the compiled index:
Bash

uv run cli/keyword_search_cli.py search "brave"

Sample Output:
Plaintext

Searching for: brave
1. Mohawk, (ID: 2054)
2. Dark Passage, (ID: 2577)
3. Tuck Everlasting, (ID: 4101)
4. Shake, Rattle & Roll, (ID: 4104)
5. How to Steal a Million, (ID: 2065)

Roadmap & Potential Enhancements

    TF-IDF & BM25 Relevance Scoring: Rank results by term frequency and inverse document frequency rather than raw document ID sequence.

    Boolean Query Logic: Implement AND, OR, and NOT set operations across multiple query terms.

    Positional Indexing: Track word positions within documents to support exact phrase searches (e.g., "brave warrior").