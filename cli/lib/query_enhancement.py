# lib/query_enhancement.py

import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def correct_spelling(query: str) -> str:
    """
    Uses OpenRouter LLM to fix spelling and typos in a search query.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a search query spell checker. Fix any spelling errors or typos "
                "in the user's search query. Respond ONLY with the corrected query text, "
                "with no explanation, preamble, or markdown formatting."
            ),
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    corrected_query = response.choices[0].message.content
    return corrected_query.strip() if corrected_query else query


def rewrite_query(query: str) -> str:
    """
    Uses OpenRouter LLM to rewrite vague queries into specific, searchable terms.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    rewritten = response.choices[0].message.content
    return rewritten.strip() if rewritten else query


def expand_query(query: str) -> str:
    """
    Uses OpenRouter LLM to expand search queries with related terms.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    expanded_terms = response.choices[0].message.content
    if not expanded_terms:
        return query

    expanded_terms = expanded_terms.strip()

    if query.lower() in expanded_terms.lower():
        return expanded_terms
    else:
        return f"{query} {expanded_terms}"


def rerank_individual(query: str, docs: list[dict]) -> list[dict]:
    """
    Re-ranks documents by asking the LLM to rate each document from 0-10 based on relevance.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    reranked_docs = []

    for i, doc in enumerate(docs):
        if i > 0:
            time.sleep(3)

        doc_text = doc.get("document") or doc.get("description", "")
        title = doc.get("title", "")

        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {doc_text}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]

        score = 0.0
        for _ in range(2):
            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=messages,
                )
                raw_text = response.choices[0].message.content or ""
                match = re.search(r"(\d+(?:\.\d+)?)", raw_text)
                if match:
                    val = float(match.group(1))
                    if 0.0 <= val <= 10.0:
                        score = val
                        break
            except Exception:
                time.sleep(2)

        doc_copy = dict(doc)
        doc_copy["rerank_score"] = score
        reranked_docs.append(doc_copy)

    # Sort descending primarily by re-rank score, tie-breaking with RRF score
    return sorted(
        reranked_docs,
        key=lambda x: (x.get("rerank_score", 0.0), x.get("rrf_score", 0.0)),
        reverse=True,
    )