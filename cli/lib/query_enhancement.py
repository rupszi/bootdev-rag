# lib/query_enhancement.py

import json
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

    return sorted(
        reranked_docs,
        key=lambda x: (x.get("rerank_score", 0.0), x.get("rrf_score", 0.0)),
        reverse=True,
    )


def rerank_batch(query: str, docs: list[dict]) -> list[dict]:
    """
    Re-ranks a list of documents in a single LLM call by requesting an ordered JSON array of IDs.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    doc_list_items = []
    for doc in docs:
        doc_id = doc.get("id")
        title = doc.get("title", "")
        description = doc.get("document") or doc.get("description", "")
        doc_list_items.append(f"\nID: {doc_id}\nTitle: {title}\nDescription: {description}\n")

    doc_list_str = "".join(doc_list_items)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    ranked_ids = []
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            cleaned_content = re.sub(r"```(?:json)?", "", content).strip("` \n\r")
            match = re.search(r"\[[\d\s,]+\]", cleaned_content)
            if match:
                ranked_ids = json.loads(match.group(0))
                if isinstance(ranked_ids, list) and len(ranked_ids) > 0:
                    break
        except Exception:
            time.sleep(2)

    rank_map = {doc_id: idx + 1 for idx, doc_id in enumerate(ranked_ids)}
    default_rank = len(docs) + 999

    reranked_docs = []
    for doc in docs:
        doc_copy = dict(doc)
        doc_copy["rerank_rank"] = rank_map.get(doc.get("id"), default_rank)
        reranked_docs.append(doc_copy)

    return sorted(
        reranked_docs,
        key=lambda x: (x.get("rerank_rank", default_rank), -x.get("rrf_score", 0.0)),
    )