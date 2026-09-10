# cli/evaluation_cli.py

import argparse
import json
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    movies = load_movies()
    searcher = HybridSearch(movies)

    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    print(f"k={limit}\n")

    test_cases = golden_dataset.get("test_cases", [])

    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = searcher.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [res["title"] for res in results]

        relevant_count = sum(1 for title in retrieved_titles if title in relevant_docs)
        precision = relevant_count / limit if limit > 0 else 0.0

        relevant_retrieved = relevant_count
        total_relevant = len(relevant_docs)
        recall = relevant_retrieved / total_relevant

        f1 = 2 * (precision * recall) / (precision + recall)

        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_docs)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}\n")


if __name__ == "__main__":
    main()