Building a Fast Keyword Search Engine from Scratch

This project implements a lightweight Inverted Index Search Engine in Python. Instead of scanning through thousands of raw JSON files every time a user types a search query, it processes raw data once, builds an index, and saves it to disk for near-instant retrieval.
What Problem Does This Solve?

Imagine you have a library containing 10,000 movie descriptions, and you want to find every movie mentioning the word "hero".

    The Naive Way (Sequential Scan): You open book #1, read every word, check for "hero", then open book #2, read every word, check for "hero"... all the way to book #10,000. This is slow and scales poorly as data grows.

    The Smart Way (Inverted Index): You flip to the index at the back of a textbook. You look up the word "hero" and immediately see a list of page numbers where it appears: [Page 12, Page 405, Page 892]. You skip straight to those pages.

This project builds that exact "back-of-the-book index" for movie titles and descriptions.
Pipeline Overview

                      ┌──────────────────────────────────────┐
                      │    RAW DATA INGESTION & PARSING      │
                      │  movies.json  +  stopwords.txt       │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    TEXT NORMALIZATION & TOKENIZATION │
                      │ Strip Punctuation -> Lowercase      │
                      │ Drop Stopwords   -> Stem Roots      │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │   BUILD INVERTED INDEX & PERSIST     │
                      │ Term -> Set[Doc IDs] & Doc ID -> Metadata│
                      │ Serialized into cache/*.pkl via Pickle│
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │        CLI SEARCH EXECUTION          │
                      │ Load cache/*.pkl -> Normalize Query  │
                      │ O(1) Set Lookup -> Sort IDs -> Print │
                      └──────────────────────────────────────┘

Step-by-Step Architecture Walkthrough
1. Data Normalization (tokenize_text)

Raw text is noisy. A search for "brave" shouldn't fail just because the document contains "Brave!", "bravely", or "Brave.".

The tokenization pipeline runs text through four distinct filters:

Raw Input:   "The brave knight fought bravely!"
1. Lowercase: "the brave knight fought bravely!"
2. No Punc:  "the brave knight fought bravely"
3. No Noise: ["brave", "knight", "fought", "bravely"]  ('the' removed via stopwords.txt)
4. Stemmed:  ["brav", "knight", "fought", "brav"]    (reduced via PorterStemmer)

Why stem words?
By reducing words to their core stems (brave → brav, bravely → brav), queries match documents regardless of grammatical suffixes.
2. The Inverted Index (InvertedIndex)

The class holds two key data structures:
self.index: dict[str, set[int]]

Maps stemmed terms to set of document IDs containing them.
Python

{
    "brav": {2054, 2577, 4101, 4104},
    "space": {101, 402, 988},
    "knight": {2054, 8812}
}

self.docmap: dict[int, dict]

Maps document IDs directly back to full movie payload metadata.
Python

{
    2054: {"id": 2054, "title": "Mohawk", "description": "A brave warrior..."},
    4101: {"id": 4101, "title": "Tuck Everlasting", "description": "A story about..."}
}

3. Disk Persistence (save & load)

Re-building an index on every terminal search command wastes time.

    build Command: Converts raw JSON into memory maps, then uses Python's pickle module to freeze these structures into binary files inside cache/:

        cache/index.pkl

        cache/docmap.pkl

    search Command: Reads binary .pkl files directly into memory in milliseconds, bypassing the raw file parsing and tokenization passes entirely.

4. Query Resolution & Deterministic Execution

When running uv run cli/keyword_search_cli.py search "brave":

    Tokenize Query: "brave" → ["brav"].

    Lookup Document Sets: idx.get_documents("brav") returns {2054, 2577, 4101, 4104}.

    Sort Document IDs: Python set iteration is non-deterministic (unordered across runs). Sorting IDs via sorted(matching_doc_ids) guarantees consistent, deterministic output across test suites and platforms.

    Hydrate Results: The first 5 document IDs are mapped back to their movie titles using idx.docmap.

Usage Guide & Verification
Build the Index

Parses raw datasets, tokenizes terms, and caches binary files to disk.
Bash

uv run cli/keyword_search_cli.py build

Search Keywords

Loads cached binary indexes and prints the top 5 matching results.
Bash

uv run cli/keyword_search_cli.py search "brave"

Example Output:
Plaintext

Searching for: brave
1. Mohawk, (ID: 2054)
2. Dark Passage, (ID: 2577)
3. Tuck Everlasting, (ID: 4101)
4. Shake, Rattle & Roll, (ID: 4104)
5. How to Steal a Million, (ID: 2065)

Want to add TF-IDF scoring so results are ranked by relevance instead of doc ID?