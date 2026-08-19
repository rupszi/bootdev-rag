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
    embed_parser = subparsers.add_parser("embed_text", help="Embed the provided text.")
    embed_parser.add_argument(
        "text", type=str, help="The text string to generate a vector embedding for"
    )

    # Command 3: 'verify_embeddings' - Builds or loads vectors for all movies
    subparsers.add_parser(
        "verify_embeddings", help="Verify document embeddings matrix shape"
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()