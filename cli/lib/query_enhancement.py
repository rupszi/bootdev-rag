# lib/query_enhancement.py

import os
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from openai import OpenAI


def correct_spelling(query: str) -> str:
    """
    Uses OpenRouter LLM to fix spelling and typos in a search query.
    Returns only the corrected query string.
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
    
    rewritten_query = response.choices[0].message.content
    return rewritten_query.strip() if rewritten_query else query

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

    # If the LLM output doesn't already contain the original query, append it
    if query.lower() in expanded_terms.lower():
        return expanded_terms
    else:
        return f"{query} {expanded_terms}"