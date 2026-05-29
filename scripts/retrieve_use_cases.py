import argparse

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_COLLECTION = "use_cases"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _build_query_text(query: str) -> str:
    return query.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve similar use cases from ChromaDB.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text description of a new operational use case.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter new use case description: ").strip()
    if not query:
        raise ValueError("Query text cannot be empty.")

    model = SentenceTransformer(args.model)
    query_embedding = model.encode([_build_query_text(query)])

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(args.collection)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=args.top_k,
        include=["metadatas", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    rows = []
    for metadata, distance in zip(metadatas, distances):
        similarity = 1 - float(distance)
        rows.append(
            {
                "ID": metadata.get("id"),
                "Title": metadata.get("title"),
                "Similarity": round(similarity, 4),
                "Distance": round(float(distance), 4),
                "Description": metadata.get("description"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results found.")
        return

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()