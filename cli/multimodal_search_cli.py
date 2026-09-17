# cli/multimodal_search_cli.py

import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


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

    # Subcommand: image_search
    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search for matching movies using an image query",
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to the input query image"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)
            for idx, res in enumerate(results, 1):
                sim_score = f"{res['similarity']:.3f}"
                print(f"{idx}. {res['title']} (similarity: {sim_score})")
                print(f"   {res['description']}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()