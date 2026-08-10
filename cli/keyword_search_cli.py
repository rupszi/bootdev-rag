import argparse
import json
import string
from nltk.stem import PorterStemmer
from typing import List
import pickle
import os

def load_movies() -> List[dict]:
    with open("data/movies.json") as f:
        data = json.load(f)
    return data["movies"]

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[str, dict] = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term) -> List[int]:
         return sorted(self.index.get(term, set()))
         
    def build(self):
            movies = load_movies()
            for m in movies:
                doc_id = m["id"]
                self.docmap[doc_id] = m
                text = f"{m['title']} {m['description']}"
                self.__add_document(doc_id, text)
             
    def save(self) -> None:

        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

def build_command():    
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    if docs:
        print(f"First document for token 'merida' = {docs[0]}")
    
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
    subparsers.add_parser("build", help="Build and save the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    movies = load_movies()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            tokenized_query = tokenize_text(args.query)
            result = []
           
            for movie in movies:
                tokenized_title = tokenize_text(movie["title"])
                if any(q_token in tokenized_title for q_token in tokenized_query):
                    result.append(movie)

            for i, movie in enumerate(result[:5], start=1):
                    print(f"{i}. {movie['title']}")

        case "build":
              build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()