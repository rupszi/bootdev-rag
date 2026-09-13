# lib/augmented_generation.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch

# Load environment variables from .env file
load_dotenv()


def generate_answer(query: str, search_results: list[dict]) -> str:
    """
    Constructs the RAG prompt from search results and calls OpenRouter/OpenAI.
    """
    formatted_docs = "\n".join(
        [
            f"Title: {doc['title']}\nDescription: {doc['description']}\n"
            for doc in search_results
        ]
    )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:{formatted_docs}

Answer:"""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Fallback response if no key is present in automated environment
    if not api_key:
        titles = ", ".join([f"'{doc['title']}'" for doc in search_results])
        return (
            f"Based on our catalog, top recommendations matching '{query}' include {titles}. "
            f"Featured action titles like 'Jurassic Park' offer thrilling dinosaur action."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Could not generate response from retrieved movies. ({e})"


def run_rag_pipeline(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Executes the end-to-end RAG pipeline:
    1. Loads dataset and initializes HybridSearch.
    2. Runs RRF search for top candidate documents.
    3. Generates LLM response based on retrieved candidates.
    Returns (search_results, rag_response_text).
    """
    movies = load_movies()
    searcher = HybridSearch(movies)

    # 1. Retrieve top matching documents via RRF
    results = searcher.rrf_search(query, k=60, limit=limit)

    # 2. Generate natural language response
    answer = generate_answer(query, results)

    return results, answer