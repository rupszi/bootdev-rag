import argparse
from lib.keyword_search import (
    BM25_B,
    BM25_K1,
    InvertedIndex,
    bm25_tf_command,
    build_command,
    tokenize_single_term,
    tokenize_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and save the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single term to query")

    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Single term to query")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate the combined term frequency and inverse document frequency for a term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single term to query")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit the query results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            print(f"Searching for: {args.query}")
            matching_doc_ids = set()
            tokenized_query = tokenize_text(args.query)

            for token in tokenized_query:
                doc_ids = idx.get_documents(token)
                for doc_id in doc_ids:
                    matching_doc_ids.add(doc_id)

            result = []
            for doc_id in sorted(matching_doc_ids):
                result.append(idx.docmap[doc_id])
                if len(result) >= 5:
                    break

            for i, movie in enumerate(result, start=1):
                print(f"{i}. {movie['title']}, (ID: {movie['id']})")

        case "build":
            build_command()

        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            stemmed_term = tokenize_single_term(args.term)
            tf = idx.get_tf(args.doc_id, stemmed_term)
            print(tf)

        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            stemmed_term = tokenize_single_term(args.term)
            idf = idx.get_idf(stemmed_term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            stemmed_term = tokenize_single_term(args.term)
            tf_idf = idx.get_tfidf(args.doc_id, stemmed_term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            stemmed_term = tokenize_single_term(args.term)
            bm25_idf = idx.get_bm25_idf(stemmed_term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index not found. Please run build first.")
                return

            results = idx.bm25_search(args.query, args.limit)
            for rank, (doc_id, score) in enumerate(results, start=1):
                movie = idx.docmap[doc_id]
                print(f"{rank}. ({doc_id}) {movie['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

