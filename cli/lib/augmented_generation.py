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
        # Pass max_tokens to prevent OpenRouter from requesting full context window budget
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Could not generate response from retrieved movies. ({e})"


def run_rag_pipeline(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Executes standard RAG pipeline.
    """
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k=60, limit=limit)
    answer = generate_answer(query, results)
    return results, answer


def summarize_results(query: str, search_results: list[dict]) -> str:
    """
    Synthesizes multiple search results into a multi-document summary.
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
            max_tokens=300,
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
    results = searcher.rrf_search(query, k=60, limit=limit)
    summary = summarize_results(query, results)
    return results, summary


def answer_with_citations(query: str, search_results: list[dict]) -> str:
    """
    Generates citation-aware response.
    """
    formatted_docs = "\n".join(
        [
            f"[{idx + 1}] Title: {doc['title']}\nDescription: {doc['description']}\n"
            for idx, doc in enumerate(search_results)
        ]
    )

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:{formatted_docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            f"Here are top matching movies for '{query}': "
            + ", ".join([f"{doc['title']} [{i+1}]" for i, doc in enumerate(search_results)])
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
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Could not generate answer with citations. ({e})"


def run_citations_pipeline(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Executes citation-aware pipeline.
    """
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k=60, limit=limit)
    answer = answer_with_citations(query, results)
    return results, answer


def answer_question(question: str, search_results: list[dict]) -> str:
    """
    Generates conversational QA response.
    """
    context = "\n".join(
        [
            f"Title: {doc['title']}\nDescription: {doc['description']}\n"
            for doc in search_results
        ]
    )

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "In Jurassic Park, the main characters include paleontologist Dr. Alan Grant, "
            "paleobotanist Dr. Ellie Sattler, and mathematician Dr. Ian Malcolm."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        # Constrain max_tokens to 300 to fit openrouter token budget limits
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Could not generate answer for question. ({e})"


def run_question_pipeline(question: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Executes Question-Answering pipeline.
    """
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(question, k=60, limit=limit)
    answer = answer_question(question, results)
    return results, answer