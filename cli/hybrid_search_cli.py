# cli/hybrid_search_cli.py

import argparse
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch, min_max_normalize
from lib.query_enhancement import correct_spelling


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Weighted Search Command
    weighted_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight balance (0.0 to 1.0)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # RRF Search Command
    rrf_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF smoothing constant k")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        default=None,
        help="Query enhancement strategy (e.g. 'spell')",
    )

    # Normalize Parser
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

        case "rrf-search":
            query = args.query

            # Perform query enhancement if requested
            if args.enhance == "spell":
                query = correct_spelling(query)

            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.rrf_search(query, args.k, args.limit)
            for i, res in enumerate(results, start=1):
                bm25_rank_str = str(res['bm25_rank']) if res['bm25_rank'] is not None else "N/A"
                semantic_rank_str = str(res['semantic_rank']) if res['semantic_rank'] is not None else "N/A"

                print(f"{i}. {res['title']}")
                print(f"  RRF Score: {res['rrf_score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank_str}, Semantic Rank: {semantic_rank_str}")
                print(f"  {res['description'][:100]}...")

        case "normalize":
            normalized_scores = min_max_normalize(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()