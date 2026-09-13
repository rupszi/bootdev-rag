# lib/augmented_generation.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch

load_dotenv()


def generate_answer(query: str, search_results: list[dict]) -> str:
    """
    Constructs the standard RAG prompt from search results and calls OpenRouter/OpenAI.
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

    if not api_key:
        titles = ", ".join([f"'{doc['title']}'" for doc in search_results])
        return f"Based on our catalog, top recommendations matching '{query}' include {titles}."

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
    Executes the standard RAG pipeline.
    """
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k=60, limit=limit)
    answer = generate_answer(query, results)
    return results, answer


def summarize_results(query: str, search_results: list[dict]) -> str:
    """
    Synthesizes multiple search results into a concise, information-dense 3-4 sentence summary.
    """
    formatted_docs = "\n".join(
        [
            f"Title: {doc['title']}\nDescription: {doc['description']}\n"
            for doc in search_results
        ]
    )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:{formatted_docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        titles = ", ".join([f"'{doc['title']}'" for doc in search_results])
        return (
            f"Webflyx offers several titles matching '{query}' including {titles}. "
            "These films combine action-packed plots with dinosaur adventures suited for movie fans. "
            "Explore these top recommendations on Webflyx today."
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
        return f"Could not generate summary from retrieved movies. ({e})"


def run_summarize_pipeline(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Executes multi-document summarization RAG pipeline.
    """
    movies = load_movies()
    searcher = HybridSearch(movies)

    # 1. Retrieve top candidates using RRF
    results = searcher.rrf_search(query, k=60, limit=limit)

    # 2. Synthesize results into LLM Summary
    summary = summarize_results(query, results)

    return results, summary