import argparse
# Import our semantic search module using the 'sems' namespace alias
import lib.semantic_search as sems


def main() -> None:
    # Set up argument parser to handle command line inputs
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    # Add subcommands so the user can type 'verify', 'embed_text', etc.
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command 1: 'verify' - Checks model architecture
    subparsers.add_parser("verify", help="Verify model loading and sequence length")

    # Command 2: 'embed_text' - Converts a single user string into a vector
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embed the provided text."
    )
    embed_text_parser.add_argument(
        "text", type=str, help="The text string to generate a vector embedding for"
    )

    # Command 3: 'verify_embeddings' - Builds or loads vectors for all movies
    subparsers.add_parser(
        "verify_embeddings", help="Verify document embeddings matrix shape"
    )

    # Command 4: 'embed_query' - Generates a vector embedding for a user search query
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generates vectors for the provided text."
    )
    embed_query_parser.add_argument(
        "query", type=str, help="The text string to generate a vector embedding for"
    )

    # Command 5: 'search' - Compares the user's query vector against document vectors
    search_parser = subparsers.add_parser("search", help="Search movies semantically")

    # Required positional argument for the search phrase
    search_parser.add_argument("query", type=str, help="Search query string")

    # Optional flag to restrict maximum returned results (defaults to 5)
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    # Command 6: 'chunk' - Splits text into fixed word-size chunks
    chunk_parser = subparsers.add_parser("chunk", help="Chunk documents")

    # Required positional argument: text string to chunk
    chunk_parser.add_argument("text", type=str, help="Chunk string")

    # Optional flag to set words per chunk (defaults to 200)
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )

    # Read user input from terminal arguments
    args = parser.parse_args()

    # Match user command to the corresponding library function
    match args.command:
        case "verify":
            sems.verify_model()

        case "embed_text":
            sems.embed_text(args.text)

        case "verify_embeddings":
            sems.verify_embeddings()

        case "embed_query":
            sems.embed_query_text(args.query)

        case "search":
            # Pass terminal query and optional limit into the search CLI driver wrapper
            sems.search_cli(args.query, args.limit)

        case "chunk":
            # Pass text and chunk_size parameter to sems.chunk
            sems.chunk(args.text, args.chunk_size)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()