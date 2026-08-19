import argparse
# Import helper functions from local semantic search module
from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main() -> None:
    # Initialize the top-level argument parser
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    # Create subparsers for CLI subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: 'verify'
    subparsers.add_parser("verify", help="Verify model loading and sequence length")

    # Subcommand: 'embed_text'
    embed_parser = subparsers.add_parser("embed_text", help="Embed the provided text.")
    embed_parser.add_argument("text", type=str, help="The text string to generate an embedding for")

    # Subcommand: 'verify_embeddings'
    subparsers.add_parser("verify_embeddings", help="Verify document embeddings matrix shape")

    # Parse CLI arguments
    args = parser.parse_args()

    # Route execution based on parsed command
    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()