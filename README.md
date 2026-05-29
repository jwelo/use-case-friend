# use-case-friend

Semantic retrieval system for HTX operational AI use cases. Given a new use case description, it finds the most semantically similar prior submissions so analysts can spot duplicates, capability overlap, and reusable prior work.

This repo is the **Python retrieval** half of a two-component, human-in-the-loop workflow. The other half is an offline government chatbot (used manually) that turns a vague problem description into a structured analysis. An analyst then pastes that structured output into this system for retrieval.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

First run will download the `all-MiniLM-L6-v2` embedding model (~90 MB). Everything else runs locally — no API keys, no network calls at query time.

## Usage

### 1. Embed the use case dataset

Source data lives in `data/Sample Use Cases.xlsx` and must have `ID`, `Title`, `Description` columns.

```bash
python scripts/embed_use_cases.py                 # embed (upserts by ID)
python scripts/embed_use_cases.py --reset         # drop and recreate the collection
python scripts/embed_use_cases.py --sheet "Name"  # pick a non-default sheet
python scripts/embed_use_cases.py --excel path/to/other.xlsx
```

Embeddings are written to `chroma_db/` (a local ChromaDB persistent store). Re-running without `--reset` upserts by `ID`, so editing the spreadsheet and re-embedding is safe.

### 2. Retrieve similar use cases

```bash
python scripts/retrieve_use_cases.py --query "Officers spend hours transcribing witness statements" --top-k 5
python scripts/retrieve_use_cases.py                  # prompts interactively
```

Output is a pandas table with `ID`, `Title`, `Similarity` (1 - cosine distance), `Distance`, and `Description` for the top-k matches.

### Recommended workflow

For best retrieval quality, don't paste a raw problem description. Use the offline chatbot first to produce structured sections:

```
## Problem Summary
## Root Causes
## Capabilities Needed   <-- most load-bearing for semantic search
## Technology Domains
## Possible Non-AI Alternatives
## Stakeholders
## Suggested Follow-Up Questions
```

Then paste the structured text (or just the `Capabilities Needed` block) as the `--query`.

## How it works

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — lightweight, local, offline-capable.
- **Vector store:** ChromaDB persistent client at `./chroma_db`, collection `use_cases`, cosine distance.
- **Document format:** each row is embedded as `Title: {title}\nDescription: {description}`.
- **Metadata:** each vector carries `id`, `title`, `description` so the retriever can render results without re-reading the Excel.

The collection name, model, store path, and metadata schema are duplicated between `embed_use_cases.py` and `retrieve_use_cases.py` and must stay in sync.

## Project status

Current priority: embedding pipeline → retrieval → retrieval quality evaluation. No UI yet.

Out of scope for now: autonomous agents, LangGraph, multi-agent orchestration, fine-tuning, hosted embedding APIs. Streamlit or Teams integration may come later.
