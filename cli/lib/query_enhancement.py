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