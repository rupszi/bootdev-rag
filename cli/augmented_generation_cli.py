# cli/augmented_generation_cli.py

import argparse
from lib.augmented_generation import (
    run_rag_pipeline,
    run_summarize_pipeline,
    run_citations_pipeline,
    run_question_pipeline,
)


def main() -> None:
    # Initialize main argument parser
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Subcommand: rag ---
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # --- Subcommand: summarize ---
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

    # --- Subcommand: citations ---
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

    # --- Subcommand: question ---
    question_parser = subparsers.add_parser(
        "question", help="Answer user question conversationally based on retrieved movies"
    )
    # Required positional argument for the user's question
    question_parser.add_argument("question", type=str, help="Question to answer")
    # Optional limit argument (defaults to 5)
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to retrieve for context",
    )

    # Parse command line inputs
    args = parser.parse_args()

    # Route execution based on command
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

        case "question":
            question = args.question
            limit = args.limit

            # Run question answering pipeline
            results, answer = run_question_pipeline(question, limit=limit)

            # Output results matching expected format
            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()