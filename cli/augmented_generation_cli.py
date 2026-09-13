# cli/augmented_generation_cli.py

import argparse
from lib.augmented_generation import run_rag_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            # Run RAG pipeline
            results, rag_response = run_rag_pipeline(query, limit=5)

            # Print formatted output matching test expectations
            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")

            print("\nRAG Response:")
            print(rag_response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()