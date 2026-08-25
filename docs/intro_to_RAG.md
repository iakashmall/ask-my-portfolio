# RAG

A minimal retrieval-augmented generation example: local sentence-transformer embeddings for retrieval, Claude for grounded generation.

## Pipeline

1. **Chunking** — documents split into retrievable units (`chunk_text`)
2. **Embedding** — `all-MiniLM-L6-v2`, 384-dim, runs locally
3. **Retrieval** — cosine similarity (dot product on normalized vectors), top-k
4. **Generation** — Claude answers using only the retrieved context

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install numpy sentence-transformers anthropic
```

Set your API key:

```bash
set ANTHROPIC_API_KEY=sk-ant-...        # Windows
export ANTHROPIC_API_KEY=sk-ant-...     # macOS/Linux
```

## Run

```bash
python trial.py
```

## Notes

The knowledge base in `trial.py` is a hardcoded list of sample strings — replace `DOCUMENTS` with your own corpus.
