# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the local venv before running scripts (Windows bash):

```
source .venv/Scripts/activate
pip install -r requirements.txt
```

Embed use cases from the Excel source into a persistent ChromaDB collection:

```
python scripts/embed_use_cases.py                 # uses data/Sample Use Cases.xlsx
python scripts/embed_use_cases.py --reset         # drop and recreate the collection
python scripts/embed_use_cases.py --sheet "Name"  # pick a non-default sheet
```

Retrieve similar use cases (interactive if `--query` is omitted):

```
python scripts/retrieve_use_cases.py --query "..." --top-k 5
```

There is no test suite, linter, or build step configured.

## Architecture

Two-stage RAG-style pipeline over a small Excel dataset of operational use cases:

1. **Embed stage** (`scripts/embed_use_cases.py`) — reads `data/Sample Use Cases.xlsx`, requires columns `ID`, `Title`, `Description`, formats each row as `Title: ...\nDescription: ...`, encodes with SentenceTransformer (`all-MiniLM-L6-v2` by default), and **upserts** into a `chromadb.PersistentClient(path="chroma_db")` collection named `use_cases` configured with cosine distance (`hnsw:space: cosine`). Re-running without `--reset` updates existing IDs in place.

2. **Retrieve stage** (`scripts/retrieve_use_cases.py`) — opens the same persistent collection, encodes the query with the same model, runs `collection.query` with `include=["metadatas", "distances"]`, and converts cosine distance to similarity via `1 - distance` before printing a pandas table.

Key coupling points to preserve when changing either script:
- The collection name, model name, and `chroma_db` path must match across both scripts.
- The collection's distance metric (`cosine`) is set at create time; changing it requires `--reset`.
- Metadata schema (`id`, `title`, `description`) is what the retriever reads back — keep these keys in sync.

The `chroma_db/` directory is the on-disk vector store (created on first embed run); deleting it forces a full re-embed.
