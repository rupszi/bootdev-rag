# cli/hybrid_search_cli.py

import argparse
from lib.keyword_search import load_movies
from lib.hybrid_search import HybridSearch, min_max_normalize
from lib.query_enhancement import (
    correct_spelling,
    rewrite_query,
    expand_query,
    rerank_individual,
    rerank_batch,
    rerank_cross_encoder,
)


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
        choices=["spell", "rewrite", "expand"],
        default=None,
        help="Query enhancement strategy ('spell', 'rewrite', or 'expand')",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        default=None,
        help="Re-ranking method strategy ('individual', 'batch', or 'cross_encoder')",
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
            original_query = args.query
            search_query = original_query

            # Pipeline Step 1: Log original input query
            print(f"\n--- [DEBUG] Pipeline Step 1: Original Query ---")
            print(f"Query: '{original_query}'")

            # Pipeline Step 2: Query Enhancement stage
            print(f"\n--- [DEBUG] Pipeline Step 2: Query Enhancement ---")
            if args.enhance == "spell":
                search_query = correct_spelling(original_query)
                print(f"Enhanced query (spell): '{original_query}' -> '{search_query}'")
            elif args.enhance == "rewrite":
                search_query = rewrite_query(original_query)
                print(f"Enhanced query (rewrite): '{original_query}' -> '{search_query}'")
            elif args.enhance == "expand":
                search_query = expand_query(original_query)
                print(f"Enhanced query (expand): '{original_query}' -> '{search_query}'")
            else:
                print(f"Enhancement Strategy: None (Using raw query: '{search_query}')")

            movies = load_movies()
            searcher = HybridSearch(movies)

            # Fetch candidate window (5x limit when re-ranking is requested)
            fetch_limit = args.limit * 5 if args.rerank_method in ("individual", "batch", "cross_encoder") else args.limit
            results = searcher.rrf_search(search_query, args.k, fetch_limit)

            # Pipeline Step 3: Candidate RRF Pool
            print(f"\n--- [DEBUG] Pipeline Step 3: Top RRF Candidates ({len(results)} candidates fetched) ---")
            for idx, res in enumerate(results[:25], start=1):
                bm25_rank_str = str(res['bm25_rank']) if res['bm25_rank'] is not None else "N/A"
                semantic_rank_str = str(res['semantic_rank']) if res['semantic_rank'] is not None else "N/A"
                print(
                    f"  {idx}. {res['title']} "
                    f"(RRF Score: {res['rrf_score']:.4f}, "
                    f"BM25 Rank: {bm25_rank_str}, "
                    f"Sem Rank: {semantic_rank_str})"
                )

            # Apply Re-Ranking if specified
            if args.rerank_method == "individual":
                print(f"\nRe-ranking top {len(results)} results using individual method...")
                results = rerank_individual(search_query, results)
                results = results[: args.limit]
            elif args.rerank_method == "batch":
                print(f"\nRe-ranking top {len(results)} results using batch method...")
                results = rerank_batch(search_query, results)
                results = results[: args.limit]
            elif args.rerank_method == "cross_encoder":
                print(f"\nRe-ranking top {len(results)} results using cross_encoder method...")
                results = rerank_cross_encoder(search_query, results)
                results = results[: args.limit]

            # Pipeline Step 4: Final Output
            print(f"\n--- [DEBUG] Pipeline Step 4: Final Top {len(results)} Results ---")
            print(f"Reciprocal Rank Fusion Results for '{search_query}' (k={args.k}):\n")

            for i, res in enumerate(results, start=1):
                bm25_rank_str = str(res['bm25_rank']) if res['bm25_rank'] is not None else "N/A"
                semantic_rank_str = str(res['semantic_rank']) if res['semantic_rank'] is not None else "N/A"

                print(f"{i}. {res['title']}")
                if "rerank_score" in res:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                elif "rerank_rank" in res:
                    print(f"   Re-rank Rank: {res['rerank_rank']}")
                elif "cross_encoder_score" in res:
                    print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {res['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {semantic_rank_str}")
                print(f"   {res['description'][:100]}...\n")

        case "normalize":
            normalized_scores = min_max_normalize(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()