import argparse
# from ast import For, Pass, Return, Store
import json
# from operator import index
import string
# from tkinter.filedialog import Open
from nltk.stem import PorterStemmer
from typing import List


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[str, dict] = {}



    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        
        
        # Tokenize text.
        # For each token, ensure a set exists in self.index.
        # Add doc_id to that set.
        # Which dictionary method creates a default value when a key is missing?

    def get_documents(self, term):

        # Retrieve the term’s set, using an empty set when absent.
        # Return its IDs sorted in ascending order.

    def build(load_movies: list):


        # Remember every instance method needs self.
        # The lesson says load_movies() is called inside this method, not passed as a parameter.
        # For each movie:

        # Store the full movie in docmap under its ID.
        # Combine its title and description.
        # Pass its ID and combined text to __add_document.

    def save():

        # Create the cache directory.
        # Open each destination in binary-write mode.
        # Serialize each dictionary with pickle.dump.

    def build_command():





def tokenize_text(text: str) -> List[str]:
    stemmer = PorterStemmer()
    with open("data/stopwords.txt") as f:
            stopwords = set(f.read().translate(str.maketrans('', '', string.punctuation)).lower().splitlines())
    punc_table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(punc_table).lower()
    return [stemmer.stem(w) for w in clean_text.split() if w not in stopwords]



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # stemmer = PorterStemmer()

    with open("data/movies.json") as f:
        movies = json.load(f)

    # with open("data/stopwords.txt") as f:
    #     stopwords = set(f.read().translate(str.maketrans('', '', string.punctuation)).lower().splitlines())
        

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = []
            # punc_table = str.maketrans("", "", string.punctuation)
            # clean_text = args.query.translate(punc_table).lower()
            tokenized_query = [word for word in clean_text.split() if word not in stopwords]
            tokenized_query = [stemmer.stem(word) for word in tokenized_query]

            for movie in movies["movies"]:
                clean_title = movie["title"].translate(punc_table).lower()
                tokenized_title = [word for word in clean_title.split() if word not in stopwords]
                tokenized_title = [stemmer.stem(word) for word in tokenized_title]
                if any(q_token in t_token for q_token in tokenized_query for t_token in tokenized_title):
                    result.append(movie)

            for i, movie in enumerate(result[:5], start=1):
                    print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()