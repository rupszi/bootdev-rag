# cli/augmented_generation_cli.py

import argparse
from lib.augmented_generation import (
    run_rag_pipeline,
    run_summarize_pipeline,
    run_citations_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # RAG Command
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # Summarize Command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Synthesize and summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize",
    )

    # Citations Command
    citations_parser = subparsers.add_parser(
        "citations", help="Answer query with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citation answer")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to evaluate and cite",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results, rag_response = run_rag_pipeline(query, limit=5)

            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")

            print("\nRAG Response:")
            print(rag_response)

        case "summarize":
            query = args.query
            limit = args.limit
            results, summary = run_summarize_pipeline(query, limit=limit)

            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            limit = args.limit
            results, citation_answer = run_citations_pipeline(query, limit=limit)

            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")

            print("\nLLM Answer:")
            print(citation_answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()