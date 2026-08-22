import argparse
import json
import lib.semantic_search as sems


def main() -> None:
    # Set up argument parser to handle command line inputs
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    # Add subcommands so the user can type 'verify', 'embed_text', etc.
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Command 1: 'verify' - Checks model architecture
    subparsers.add_parser(
        "verify", help="Verify model loading and sequence length"
    )

    # Command 2: 'embed_text' - Converts a single user string into a vector
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embed the provided text."
    )
    embed_text_parser.add_argument(
        "text",
        type=str,
        help="The text string to generate a vector embedding for",
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
        "query",
        type=str,
        help="The text string to generate a vector embedding for",
    )

    # Command 5: 'search' - Compares the user's query vector against document vectors
    search_parser = subparsers.add_parser(
        "search", help="Search movies semantically"
    )
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    # Command 6: 'chunk' - Splits text into fixed word-size chunks
    chunk_parser = subparsers.add_parser("chunk", help="Chunk documents")
    chunk_parser.add_argument("text", type=str, help="Chunk string")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlap words per chunk (default: 0)",
    )

    # Command 7: 'semantic_chunk' - Splits text into semantic sentence chunks
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantically chunk documents by sentence"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Chunk string")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Number of sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentence overlaps per chunk (default: 0)",
    )

    # Command 8: 'embed_chunks' - Generates chunked embeddings for all movie descriptions
    subparsers.add_parser(
        "embed_chunks", help="Generate and cache chunked embeddings"
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
            sems.search_cli(args.query, args.limit)

        case "chunk":
            sems.chunk(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            sems.semantic_chunk(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            # Load documents dataset before passing to embed_chunks
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            sems.embed_chunks(documents)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()