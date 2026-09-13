# cli/multimodal_search_cli.py

import argparse
from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: verify_image_embedding
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate and print the shape of an image embedding",
    )
    verify_parser.add_argument(
        "image_path", type=str, help="Path to the image file to embed"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()