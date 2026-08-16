Here is an expanded, non-technical-friendly README designed to make the engine's purpose, importance, and technical mechanics crystal clear to both technical developers and non-technical stakeholders alike.
Fast Keyword Search Engine in Python

A lightweight, high-performance Inverted Index Search Engine built from scratch in Python. It transforms unstructured document datasets (like movie catalogs) into a lightning-fast, structured index—allowing you to search thousands of documents instantly without scanning through them line by line.
💡 What Is This Project? (Plain English Version)

Imagine you are looking for a specific topic in a 1,000-page encyclopedia.

    The Slow Way (Linear Scan): You open to page 1 and read every single word on every page until you find what you are looking for. If the book is huge, this takes forever.

    The Smart Way (Inverted Index): You flip directly to the index at the back of the book. You look up your word alphabetically, see a list of exact page numbers where that word appears, and turn directly to those pages.

This Python program builds that exact "index at the back of the book" for digital computer files.

Instead of reading through thousands of movie titles and descriptions every time you type a query, this tool reads the entire database once, builds a master map of every word and where it lives, and saves that map to disk. When you search, it jumps directly to the matching results in a fraction of a millisecond.
🎯 Why Is This Important?

In the modern digital world, data grows exponentially every day. Understanding why index-based search matters comes down to three key reasons:
1. Speed and Scalability

If you search 1,000 movies by checking each one sequentially, it takes a few milliseconds. But if you try to search 10,000,000 documents using that same sequential approach, your search will crash or take minutes to return a result. Building an inverted index shifts all the heavy processing work to a single "prep step" so that end-user searches remain instant, regardless of dataset size.
2. Smart Language Matching (It Understands Word Roots)

Computers are notoriously literal. Normally, if a movie description says "The hero fought bravely" and you search for "brave", a basic search engine won't find it because the words are spelled differently.

This search engine includes a Text Normalization Pipeline that cleans up human language. It reduces words down to their root stems (bravely → brav, running → run), ensuring you find relevant results even if you don't type the exact grammatical variation.
3. Measuring Relevance (TF-IDF Scoring)

Not all words are created equal. Words like "the", "is", or "and" appear thousands of times but tell us nothing about what a movie is about. On the other hand, a rare word like "galaxy" or "dinosaur" carries heavy meaning.

This engine implements TF-IDF (Term Frequency-Inverse Document Frequency) math to calculate how important a word is to a specific document relative to the rest of the collection, laying the foundation for modern relevance ranking.
🏗 System Architecture

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
                       │ O(1) Set Lookup -> Calculate TF-IDF  │
                       └──────────────────────────────────────┘

🛠 Architectural Walkthrough (How It Works Under the Hood)
1. Text Normalization Pipeline (tokenize_text)

Before any text is indexed or searched, it passes through four cleanup stages:

Raw Input:  "The brave knight fought bravely!"
  ├── 1. Case Fold:   "the brave knight fought bravely!"
  ├── 2. Strip Punc:  "the brave knight fought bravely"
  ├── 3. Filter Stop: ["brave", "knight", "fought", "bravely"]   (removes common noise words)
  └── 4. Stem Roots:  ["brav", "knight", "fought", "brav"]     (PorterStemmer root reduction)

2. Primary Data Structures (InvertedIndex)

The engine keeps track of documents using three specialized dictionary structures:

    Postings Index (self.index): Maps stemmed words directly to the set of document IDs containing them.
    Python

    {
        "brav": {2054, 2577, 4101, 4104},
        "space": {101, 402, 988},
        "knight": {2054, 8812}
    }

    Document Map (self.docmap): Maps unique document IDs back to the original full movie records so titles and descriptions can be displayed instantly.
    Python

    {
        2054: {"id": 2054, "title": "Mohawk", "description": "A brave warrior..."},
        4101: {"id": 4101, "title": "Tuck Everlasting", "description": "A story about..."}
    }

    Term Frequency Tracking (self.term_frequencies): Remembers how many times each word appears inside each specific document for scoring calculations.

3. Disk Serialization & Caching

    build Phase: Reads the raw JSON files, processes all text, constructs the inverted index in memory, and saves binary cache snapshots (cache/*.pkl) using Python's pickle module.

    search / Metric Phase: Reads the tiny binary cache files directly into memory in milliseconds, bypassing the need to re-read or re-parse the raw dataset ever again.

🚦 Where Are We Now? (Current Project Status)

The core search engine, storage pipeline, and statistical scoring features are fully built and working.
🌟 Implemented Features

    ✅ Full text cleaning, stop-word removal, and word stemming.

    ✅ Instant single and multi-term keyword search lookup.

    ✅ Persistent disk caching via binary pickle snapshots.

    ✅ Term Frequency (TF) tracking per document.

    ✅ Inverse Document Frequency (IDF) calculation across the entire corpus.

    ✅ TF-IDF metric calculation for individual document-term pairs.

📖 Usage Guide & CLI Commands
Installation & Environment Setup

Ensure uv is installed, then sync project dependencies:
Bash

uv sync

1. Build the Index

Parse raw input datasets (data/movies.json) and generate binary cache files in cache/:
Bash

uv run cli/keyword_search_cli.py build

2. Search for Movies

Run keyword queries against the pre-built index:
Bash

uv run cli/keyword_search_cli.py search "brave warrior"

Sample Output:
Plaintext

Searching for: brave warrior
1. Mohawk, (ID: 2054)
2. Dark Passage, (ID: 2577)
3. Tuck Everlasting, (ID: 4101)
4. Shake, Rattle & Roll, (ID: 4104)
5. How to Steal a Million, (ID: 2065)

3. Calculate Term Frequency (TF)

Find how many times a word appears in a specific document:
Bash

uv run cli/keyword_search_cli.py tf 2054 "brave"

4. Calculate Inverse Document Frequency (IDF)

Measure how rare or informative a word is across all documents:
Bash

uv run cli/keyword_search_cli.py idf "galaxy"

5. Calculate TF-IDF Score

Determine the exact mathematical relevance of a keyword to a specific document:
Bash

uv run cli/keyword_search_cli.py tfidf 2054 "brave"

🚀 Roadmap & Next Steps

    [ ] Ranked Search Results: Automatically sort multi-term search outputs by descending TF-IDF score rather than numerical document ID.

    [ ] BM25 Algorithm: Upgrade from basic TF-IDF to Okapi BM25 for advanced industrial search ranking.

    [ ] Boolean Search Support: Support logical operators like AND, OR, and NOT (e.g., space AND NOT alien).

    [ ] Positional Indexing: Store word position coordinates to support exact phrase matching (e.g., "star wars").