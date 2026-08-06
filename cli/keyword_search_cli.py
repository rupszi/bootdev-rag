import argparse
import json
import string


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/movies.json") as f:
        movies = json.load(f)
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = []
            punc_table = str.maketrans("", "", string.punctuation)
            clean_text = args.query.translate(punc_table).lower()
            tokenized_query = clean_text.split()

            for movie in movies["movies"]:
                clean_title = movie["title"].translate(punc_table).lower()
                tokenized_title = clean_title.split()
                if any(q_token in t_token for q_token in tokenized_query for t_token in tokenized_title):
                    result.append(movie)

            for i, movie in enumerate(result[:5], start=1):
                    print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()