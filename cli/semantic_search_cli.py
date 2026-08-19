import argparse
# Import helper functions from the local semantic search library module
from lib.semantic_search import embed_text, verify_model


def main() -> None:
    # Initialize the top-level argument parser for the Semantic Search CLI
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    # Create subparsers to handle distinct subcommands (e.g., 'verify', 'embed_text')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: 'verify' (requires no additional arguments)
    subparsers.add_parser("verify", help="Verify model loading and sequence length")

    # Subcommand: 'embed_text' (requires one positional string argument)
    embed_parser = subparsers.add_parser("embed_text", help="Embed the provided text.")
    
    # Define positional argument 'text' so argparse captures trailing string inputs
    embed_parser.add_argument(
        "text", 
        type=str, 
        help="The text string to generate a vector embedding for"
    )

    # Parse command-line arguments passed at execution time
    args = parser.parse_args()

    # Route execution based on the chosen subcommand
    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            # Pass the parsed positional 'text' argument to embed_text
            embed_text(args.text)

        case _:
            # Print help text if no valid subcommand is provided
            parser.print_help()


if __name__ == "__main__":
    main()