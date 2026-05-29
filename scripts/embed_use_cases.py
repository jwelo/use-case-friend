import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb


DEFAULT_EXCEL_PATH = Path("data") / "Sample Use Cases.xlsx"
DEFAULT_COLLECTION = "use_cases"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _build_text(title: str, description: str) -> str:
	title = (title or "").strip()
	description = (description or "").strip()
	if title and description:
		return f"Title: {title}\nDescription: {description}"
	return title or description


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {', '.join(missing)}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Embed use cases into ChromaDB.")
	parser.add_argument(
		"--excel",
		type=Path,
		default=DEFAULT_EXCEL_PATH,
		help="Path to the Excel file containing use cases.",
	)
	parser.add_argument(
		"--sheet",
		type=str,
		default=None,
		help="Optional sheet name. Defaults to the first sheet.",
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
	parser.add_argument(
		"--reset",
		action="store_true",
		help="Delete and recreate the collection before embedding.",
	)
	args = parser.parse_args()

	if not args.excel.exists():
		raise FileNotFoundError(f"Excel file not found: {args.excel}")

	df = pd.read_excel(args.excel, sheet_name=args.sheet)
	_validate_columns(df, ["ID", "Title", "Description"])
	df = df.dropna(subset=["ID", "Title", "Description"], how="all").copy()

	ids = df["ID"].astype(str).tolist()
	titles = df["Title"].fillna("").astype(str).tolist()
	descriptions = df["Description"].fillna("").astype(str).tolist()
	documents = [_build_text(title, description) for title, description in zip(titles, descriptions)]
	metadatas = [
		{
			"id": record_id,
			"title": title,
			"description": description,
		}
		for record_id, title, description in zip(ids, titles, descriptions)
	]

	model = SentenceTransformer(args.model)
	embeddings = model.encode(documents, show_progress_bar=True)

	client = chromadb.PersistentClient(path="chroma_db")
	if args.reset:
		try:
			client.delete_collection(args.collection)
		except ValueError:
			pass

	collection = client.get_or_create_collection(
		name=args.collection,
		metadata={"hnsw:space": "cosine"},
	)
	collection.upsert(
		ids=ids,
		documents=documents,
		embeddings=embeddings.tolist(),
		metadatas=metadatas,
	)

	print(f"Embedded {len(ids)} use cases into collection '{args.collection}'.")


if __name__ == "__main__":
	main()
