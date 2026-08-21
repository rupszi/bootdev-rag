# Advanced Search Engine: Inverted Index & Semantic Vector Retrieval in Python

A high-performance, hybrid search engine CLI platform built from scratch in Python. The system provides two complementary retrieval architectures:

1. **Fast Keyword Inverted Index Engine:** A fast lexical search engine utilizing text normalization, stem reduction, postings lists, term-frequency matrix tracking, and TF-IDF relevance metrics.
2. **Dense Vector Semantic Engine:** A neural semantic search engine powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and `numpy` matrix calculations, featuring fixed-size word sliding-window chunking and sentence-boundary regex semantic chunking with overlap context preservation.

---

## 💡 System Overview: Why Dual-Engine Architecture?

Modern information retrieval systems face a fundamental trade-off between **lexical exactness** and **conceptual understanding**:

```
                              ┌───────────────────────────────────────────────┐
                              │            RAW QUERY / DOCUMENT               │
                              └──────────────────────┬────────────────────────┘
                                                     │
                   ┌─────────────────────────────────┴─────────────────────────────────┐
                   ▼                                                                   ▼
    ┌─────────────────────────────┐                                     ┌─────────────────────────────┐
    │   LEXICAL INVERTED INDEX    │                                     │    DENSE VECTOR SEMANTIC    │
    │  (Keyword-exact matching)   │                                     │   (Neural Vibe/Context)     │
    └──────────────┬──────────────┘                                     └──────────────┬──────────────┘
                   │                                                                   │
    • Tokenize, Stopwords, Stemming                                     • SentenceTransformer Model
    • Postings Maps: Term -> Set[DocID]                                 • 384-Dimensional Embeddings
    • Fast O(1) Set Intersections                                       • Cosine Similarity Dot Product
    • Exact Model Numbers / Names                                       • Conceptual / Synonymous Queries

```

* **The Lexical Approach (Inverted Index):** Outstanding at finding exact strings, product IDs, rare proper nouns, and specific codes. However, if a user queries "feline," a purely lexical system fails if the document only contains "cat."
* **The Dense Semantic Approach (Vector Search):** Transforms text into continuous 384-dimensional vector spaces. It understands that "spacecraft" and "spaceship" occupy nearly identical vector coordinates.
* **Text Chunking Pipeline:** Addresses the transformer model sequence length bottleneck ($N_{\text{max}} = 256$ tokens) by partitioning long text documents into overlapping fixed-word or sentence-level segments without losing context at boundaries.

---

## 🏗 System Architecture & End-to-End Pipeline

```
                                    ┌──────────────────────────────────────┐
                                    │      RAW DATASETS & DOCUMENTS        │
                                    │ data/movies.json + data/stopwords.txt│
                                    └──────────────────┬───────────────────┘
                                                       │
                   ┌───────────────────────────────────┴───────────────────────────────────┐
                   │                                                                       │
                   ▼                                                                       ▼
    ┌─────────────────────────────┐                                         ┌─────────────────────────────┐
    │  TEXT NORMALIZATION ENGINE  │                                         │      CHUNKING PIPELINE      │
    ├─────────────────────────────┤                                         ├─────────────────────────────┤
    │ 1. Lowercase Folding        │                                         │ 1. Fixed Word Sliding Window│
    │ 2. Punctuation Strip        │                                         │ 2. Regex Sentence Splitter  │
    │ 3. Stopword Filtering       │                                         │ 3. Overlap Stride Control   │
    │ 4. Porter Stemmer Roots     │                                         └──────────────┬──────────────┘
    └──────────────┬──────────────┘                                                        │
                   │                                                                       │
                   ▼                                                                       ▼
    ┌─────────────────────────────┐                                         ┌─────────────────────────────┐
    │    INVERTED INDEX STORE     │                                         │   DENSE VECTOR EMBEDDINGS   │
    ├─────────────────────────────┤                                         ├─────────────────────────────┤
    │ • Postings: Term -> Set     │                                         │ • all-MiniLM-L6-v2 Encoder  │
    │ • Term Frequency Matrix     │                                         │ • Matrix: N x 384 Float32   │
    │ • TF-IDF Metric Computation │                                         │ • Disk Cache: *.npy Vector  │
    │ • Pickle Persistence (.pkl) │                                         └──────────────┬──────────────┘
    └──────────────┬──────────────┘                                                        │
                   │                                                                       │
                   └───────────────────────────────────┬───────────────────────────────────┘
                                                       │
                                                       ▼
                                    ┌──────────────────────────────────────┐
                                    │         CLI SEARCH DRIVER            │
                                    │ keyword_search_cli | semantic_search │
                                    └──────────────────────────────────────┘

```

---

## 🛠 In-Depth Architectural Components

### 1. Lexical Keyword Inverted Index (`cli/lib/keyword_search.py`)

#### Text Normalization Pipeline

Before text enters the postings index, it undergoes deterministic transformation:

1. **Lowercasing:** Converts strings to lowercase to enforce case-insensitive matching.
2. **Punctuation Stripping:** Removes non-alphanumeric noise using string translation tables.
3. **Stopword Filtering:** Eliminates high-frequency, low-information tokens (e.g., `"the"`, `"and"`, `"is"`) provided in `data/stopwords.txt`.
4. **Porter Stemming:** Reduces inflectional word variants down to unified root stems (e.g., `"running"`, `"ran"`, `"runs"` $\rightarrow$ `"run"`).

#### Primary Memory Structures

* **Postings List (`self.index`):** A dictionary mapping stemmed terms directly to a Python `set` of document IDs:
```python
{
    "brav": {2054, 2577, 4101},
    "space": {101, 402, 988}
}

```


* **Document Map (`self.docmap`):** An $O(1)$ lookup mapping document IDs back to their full JSON metadata records.
* **Term Frequencies (`self.term_frequencies`):** Nested mapping tracking term occurrences per document: `dict[int, dict[str, int]]`.

#### TF-IDF Mathematical Scoring

To rank relevance, the system calculates the Term Frequency-Inverse Document Frequency weight:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

$$\text{IDF}(t, D) = \log_e \left( \frac{\vert{}D\vert{}}{\vert{}\{d \in D : t \in d\}\vert{}} \right)$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

### 2. Dense Semantic Vector Search (`cli/lib/semantic_search.py`)

#### Vector Embedding Generation

Utilizes the `all-MiniLM-L6-v2` SentenceTransformer architecture to project input text into a high-dimensional vector space ($\mathbb{R}^{384}$).

* Input sentences are converted to dense floating-point vector arrays.
* Pre-computed embeddings are serialized to disk as uncompressed NumPy matrix caches (`cache/movie_embeddings.npy`) to ensure instant cold-start execution.

#### Vector Similarity Metric (Cosine Similarity)

Determines the angular proximity between two $n$-dimensional vectors $\mathbf{u}$ and $\mathbf{v}$, independent of magnitude:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

When a query vector is computed, the engine calculates cosine similarity against all document row vectors in the cached embedding matrix, returning the top $N$ closest semantic matches ranked by descending score.

---

### 3. Document Chunking Engine

Long-form documents exceed vector context windows and obscure granular details. The library provides two distinct chunking strategies to break down text while preserving context through sliding window overlaps.

```
SLIDING WINDOW CHUNKING WITH OVERLAP (chunk_size=4, overlap=2, stride=2):

Words:   [ W1   W2   W3   W4 ]  W5   W6   W7   W8
         └────── Chunk 1 ─────┘
                     [ W3   W4   W5   W6 ]
                     └────── Chunk 2 ─────┘
                                 [ W5   W6   W7   W8 ]
                                 └────── Chunk 3 ─────┘
                                 ▲───────▲ Overlap Region

```

#### Strategy A: Fixed-Word Window Chunking (`chunk`)

Splits text into words on whitespace and slides a window defined by `chunk_size` and `overlap`.

* **Stride Calculation:** $\text{stride} = \text{chunk\_size} - \text{overlap}$
* **Pointer Loop:** Advances pointer $i$ by $\text{stride}$ on each step.
* **Termination Rule:** Halts when $i + \text{chunk\_size} \ge N_{\text{words}}$ to prevent duplicate tail slices.

#### Strategy B: Semantic Sentence Boundary Chunking (`semantic_chunk`)

Preserves natural grammatical boundary structures instead of cutting words arbitrarily mid-thought.

* **Regex Sentence Splitter:** `r"(?<=[.!?])\s+"`
Uses a *positive lookbehind* `(?<=[.!?])` to match whitespace `\s+` immediately following a period, exclamation point, or question mark. This splits text at sentence boundaries without stripping terminal punctuation from sentence strings.
* **Sentence Windowing:** Groups up to `max_chunk_size` sentences into a unified chunk string with `overlap` sentence retention between successive chunks.

---

## 🛠 Environment Setup & Installation

Ensure `uv` is installed on your machine. Sync dependencies from the project root:

```bash
uv sync

```

---

## 📖 CLI Operations & Usage Guide

The project exposes two CLI interfaces: `keyword_search_cli.py` for lexical inverted index operations and `semantic_search_cli.py` for vector semantic operations and chunking.

### Section A: Keyword Inverted Index Engine

#### 1. Build the Inverted Index Cache

Parses raw datasets (`data/movies.json`), processes text normalization, and serializes index state to `cache/*.pkl`:

```bash
uv run cli/keyword_search_cli.py build

```

#### 2. Search Keyword Index

Runs an instant lexical query over the pre-built postings index:

```bash
uv run cli/keyword_search_cli.py search "brave warrior"

```

#### 3. Compute Term Frequency (TF)

Displays how many times a normalized stem term appears within a specific document ID:

```bash
uv run cli/keyword_search_cli.py tf 2054 "brave"

```

#### 4. Compute Inverse Document Frequency (IDF)

Calculates the global rarity weight of a term across the entire corpus:

```bash
uv run cli/keyword_search_cli.py idf "galaxy"

```

#### 5. Calculate Full TF-IDF Metric

Computes the exact mathematical TF-IDF score for a document-term pair:

```bash
uv run cli/keyword_search_cli.py tfidf 2054 "brave"

```

---

### Section B: Dense Vector Semantic Engine & Chunking

#### 1. Verify SentenceTransformer Model Architecture

Inspects model dimensions and sequence length constraints:

```bash
uv run cli/semantic_search_cli.py verify

```

#### 2. Verify Document Embeddings Matrix

Loads or builds vector matrix cache for the movie dataset and displays shape metadata:

```bash
uv run cli/semantic_search_cli.py verify_embeddings

```

#### 3. Perform Vector Semantic Search

Executes a dense vector similarity query, matching meaning rather than exact keywords:

```bash
uv run cli/semantic_search_cli.py search "space exploration adventure" --limit 5

```

#### 4. Fixed-Word Window Chunking

Splits arbitrary text strings into word chunks with optional sliding overlap:

```bash
uv run cli/semantic_search_cli.py chunk "The quick brown fox jumps over the lazy dog near the riverbank." --chunk-size 4 --overlap 2

```

*Sample Output:*

```text
Chunking 68 characters
1. The quick brown fox
2. brown fox jumps over
3. jumps over the lazy
4. the lazy dog near
5. dog near the riverbank.

```

#### 5. Semantic Sentence Chunking

Splits text on sentence boundaries using regex lookbehinds with sentence-level overlap:

```bash
uv run cli/semantic_search_cli.py semantic_chunk "First sentence here. Second sentence here. Third sentence here. Fourth sentence here." --max-chunk-size 2 --overlap 1

```

*Sample Output:*

```text
Semantically chunking 85 characters
1. First sentence here. Second sentence here.
2. Second sentence here. Third sentence here.
3. Third sentence here. Fourth sentence here.

```
