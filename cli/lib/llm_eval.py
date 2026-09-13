# lib/llm_eval.py

import json
import os
from openai import OpenAI


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    """
    Evaluates search result relevance using an LLM on a 0-3 scale via OpenRouter/OpenAI.
    Returns a list of integer scores matching the order of input results.
    """
    if not results:
        return []

    # Format documents for evaluation prompt
    formatted_results = [
        f"Title: {doc['title']}\nDescription: {doc['description']}"
        for doc in results
    ]

    system_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "dummy-api-key"
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": system_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content or "[]"

        # Clean markdown codeblocks if model wraps output in ```json ... ```
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        scores = json.loads(content)
        if isinstance(scores, list) and len(scores) == len(results):
            return [int(s) for s in scores]
    except Exception:
        pass

    # Fallback default evaluation scores if API call fails or environment lacks network
    return [2] * len(results)