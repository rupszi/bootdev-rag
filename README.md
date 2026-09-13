To accurately update the `README.md` file to reflect the overall state of the search engine platform up through **Chapter 10 (Augmented Generation)**, you need to expand the documentation from a dual-engine lexical/vector system to a full **Hybrid RAG (Retrieval-Augmented Generation) Pipeline**.

Here is the updated, comprehensive `README.md` file:

```markdown
# Advanced Search Engine & RAG Platform: Hybrid Retrieval, Re-Ranking & Generation

A high-performance, production-grade search and Retrieval-Augmented Generation (RAG) platform built from scratch in Python. The platform implements an end-to-end information retrieval and generation architecture—progressing from low-level inverted index data structures and dense vector embeddings to hybrid rank fusion, query expansion, cross-encoder re-ranking, LLM evaluation, and context-augmented answer generation.

---

## 💡 System Overview: Why Hybrid Search & Augmented Generation?

Modern retrieval systems face fundamental trade-offs between **lexical precision** and **conceptual understanding**. Single-strategy retrieval often falls short:
* **Lexical exactness** misses semantically equivalent terms ("feline" vs "cat").
* **Vector similarity** can hallucinate relevance or miss exact identifiers, serial numbers, or rare proper nouns.
* **Pure LLM Generation** suffers from parameter age-off and hallucinations.

This platform bridges these gaps by combining sparse lexical indexing, dense vector space representations, reciprocal rank fusion (RRF), cross-encoder re-ranking, and dynamic LLM contextual synthesis.


```

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       USER QUERY                                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
│
┌─────────────────────────┴─────────────────────────┐
▼                                                   ▼
┌─────────────────────────────┐                     ┌─────────────────────────────┐
│   LEXICAL INVERTED INDEX    │                     │    DENSE VECTOR SEMANTIC    │
│      (BM25 Matching)        │                     │   (Neural Vector Space)     │
└──────────────┬──────────────┘                     └──────────────┬──────────────┘
│                                                   │
└─────────────────────────┬─────────────────────────┘
│
▼
┌────────────────────┐
│ SCORE NORMALIZATION│
│   OR RRF FUSION    │
└──────────┬─────────┘
│
▼
┌────────────────────┐
│   CROSS-ENCODER    │
│     RE-RANKING     │
└──────────┬─────────┘
│
▼
┌────────────────────┐
│    LLM EVAL &      │
│  RAG GENERATION    │
└────────────────────┘

```

---

## 🏗 End-to-End System Architecture

1. **Lexical Indexing Engine:** Custom inverted index supporting text normalization (lowercase folding, punctuation stripping, stopword removal, Porter stemming), postings lists, TF-IDF calculation, and full BM25 scoring with configurable $k_1$ and $b$ parameters.
2. **Dense Semantic Vector Engine:** Powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and `numpy` matrix calculations. Features fixed-word sliding-window chunking and sentence-boundary regex chunking with overlap context preservation.
3. **Hybrid Search & Fusion:**
   * **Score Normalization:** Min-Max scaling for linear weighting ($w_{\text{BM25}} \cdot S_{\text{BM25}} + w_{\text{Sem}} \cdot S_{\text{Sem}}$).
   * **Reciprocal Rank Fusion (RRF):** Position-based rank aggregation ($RRF(d) = \sum \frac{1}{k + r(d)}$) to merge keyword and semantic candidate lists without calibration.
4. **Query Expansion & Rewriting:** LLM-assisted query modification (generating synonyms, alternative phrasing, and spell-corrections) prior to index retrieval.
5. **Cross-Encoder Re-Ranking:** Deep attention-based re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` to evaluate full joint query-document context.
6. **LLM Relevance Evaluation:** Automated zero-shot/few-shot grading of retrieved search context on a 0–3 relevance scale using OpenRouter LLM endpoints.
7. **Augmented Generation (RAG):** Context injection pipeline that formats search candidates into structured prompt constraints for natural language answer synthesis.

---

## 🛠 Architectural Components

### 1. Lexical Keyword Engine (`lib/keyword_search.py` & `lib/bm25.py`)
* **Text Normalization:** Case folding, punctuation removal, stopword filtering (`data/stopwords.txt`), and Porter stem reduction.
* **Postings Index:** Dict mapping root stems to document sets for fast candidate retrieval.
* **BM25 Scoring Equation:**
  $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avgdl}}\right)}$$

### 2. Semantic Vector Engine (`lib/semantic_search.py`)
* Projections in 384-dimensional continuous space using `all-MiniLM-L6-v2`.
* Pre-computed dot product matrix cache (`cache/movie_embeddings.npy`).
* Fixed-word window and regex lookbehind sentence chunking (`r"(?<=[.!?])\s+"`) with overlap retention.

### 3. Hybrid Search & Fusion Engine (`lib/hybrid_search.py`)
* **Min-Max Score Normalizer:** Transforms raw BM25 and cosine values to a $[0, 1]$ interval.
* **Reciprocal Rank Fusion (RRF):** Combines rank positions using $k=60$ smoothing constant to eliminate score distribution discrepancies between lexical and vector scores.

### 4. Cross-Encoder Re-Ranker (`lib/re_ranker.py`)
* Utilizes full joint self-attention across query and document pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* Re-evaluates top $N$ RRF candidates to resolve semantic nuances missed by bi-encoders.

### 5. LLM Evaluation & RAG Pipeline (`lib/llm_eval.py` & `cli/augmented_generation_cli.py`)
* **Evaluation (`evaluate_results`):** Sends document sets and query pairs to LLMs via OpenRouter to obtain 0–3 relevance scores.
* **RAG Generator:** Combines top candidate contexts into standard instructions to synthesize direct, natural-language responses.

---

## 🛠 Environment Setup & Installation

Ensure `uv` is installed on your machine. Sync project dependencies:

```bash
uv sync

```

Set your OpenRouter or OpenAI API key for query expansion, evaluation, and RAG features:

```bash
export OPENROUTER_API_KEY="your-api-key"

```

---

## 📖 CLI Usage Guide

The platform exposes CLI interfaces covering each stage of the retrieval pipeline:

### 1. Inverted Index & Lexical Search (`cli/keyword_search_cli.py`)

```bash
# Build index cache
uv run cli/keyword_search_cli.py build

# BM25 Search
uv run cli/keyword_search_cli.py bm25search "action adventure in space"

# Term metrics
uv run cli/keyword_search_cli.py tfidf 2054 "brave"

```

### 2. Semantic Vector Search & Chunking (`cli/semantic_search_cli.py`)

```bash
# Verify model and pre-computed embedding shapes
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings

# Semantic vector search
uv run cli/semantic_search_cli.py search "space exploration adventure" --limit 5

# Text chunking
uv run cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence. Third sentence." --max-chunk-size 2 --overlap 1

```

### 3. Hybrid Search & Fusion (`cli/hybrid_search_cli.py`)

```bash
# Normalized hybrid search with custom weighting
uv run cli/hybrid_search_cli.py search "bear in the woods" --weight-bm25 0.5 --weight-semantic 0.5

# Reciprocal Rank Fusion (RRF) search
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --k 60

# Hybrid search with cross-encoder re-ranking
uv run cli/hybrid_search_cli.py rrf-search "time travel paradox" --rerank

# Hybrid search with LLM evaluation scoring
uv run cli/hybrid_search_cli.py rrf-search "dinosaur park" --evaluate

```

### 4. Retrieval-Augmented Generation (`cli/augmented_generation_cli.py`)

```bash
# Run end-to-end RAG pipeline
uv run cli/augmented_generation_cli.py rag "movies about action and dinosaurs"

```

*Sample RAG Output:*

```text
Search Results:
- We're Back! A Dinosaur's Story
- Jurassic Park
- The Lost World
- Carnosaur
- A Sound of Thunder

RAG Response:
Webflyx offers several action-packed dinosaur movies. "Jurassic Park" and its sequel "The Lost World" follow cloned dinosaurs causing chaos on island preserves. For animated family fun, "We're Back! A Dinosaur's Story" features intelligent dinosaurs visiting modern-day New York, while "Carnosaur" and "A Sound of Thunder" offer sci-fi thriller takes on prehistoric encounters.

```

```

```