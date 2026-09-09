# cli/hybrid_search_cli.py

import argparse
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch, min_max_normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Weighted Search Command (subcommand name updated to 'weighted-search')
    weighted_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight balance (0.0 to 1.0)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # RRF Search Command
    rrf_parser = subparsers.add_parser("rrf", help="Reciprocal Rank Fusion hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF smoothing constant k")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # Add the 'normalize' parser
    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Space-separated list of float scores to normalize",
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
                print(f"  Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"  {res['description'][:100]}...")

        case "rrf":
            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.rrf_search(args.query, args.k, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")

        case "normalize":
            normalized_scores = min_max_normalize(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()