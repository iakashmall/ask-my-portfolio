# Ask My Portfolio — RAG over my own projects

A retrieval-augmented generation (RAG) assistant that answers questions about my
own engineering projects, grounded in their actual READMEs — not hallucinated.

## Why

Recruiters and interviewers often ask "what did you actually do in project X?"
This is a small, real RAG system that answers that question correctly, citing
its sources, using the same architecture pattern used in production LLM apps.

## Architecture

```
docs/*.md  →  chunk (LangChain RecursiveCharacterTextSplitter)
           →  embed (sentence-transformers, all-MiniLM-L6-v2, local, free)
           →  index (FAISS vector store, local)
           →  retrieve top-k relevant chunks for a query
           →  generate (Claude, answer constrained to retrieved context only)
```

## Stack

- **LangChain** — document loading, chunking, retrieval chain orchestration
- **FAISS** — local vector database for similarity search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API cost
- **Ollama + Llama 3.2** — free, local, open-source LLM for grounded generation (no API key, runs on your own machine)
- **Knowledge base**: READMEs from my own repos (DILLI·GRID load forecasting,
  regime-aware trading system, IEX price forecasting, outage management system)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Install Ollama** (free, local LLM runtime):
1. Download from https://ollama.com and install it.
2. Pull a small model: `ollama pull llama3.2`
3. Ollama runs a local server automatically after install — no further setup needed.

## Usage

```bash
# 1. Build the vector index (run once, or whenever docs/ changes)
python ingest.py

# 2. Ask questions
python query.py "What model architectures were benchmarked for load forecasting?"
python query.py "How does the trading system validate incoming market data?"
```

Each answer prints the retrieved source chunks and retrieval latency, so
retrieval quality is inspectable, not a black box.

## Example

```
$ python query.py "What MAPE did the DILLI-GRID project achieve?"

Q: What MAPE did the DILLI-GRID project achieve?
A: The DILLI-GRID project achieved 2.89% MAPE (R² 0.975) after fixing
   tree-model extrapolation via a relative-target formulation, cutting
   peak-hour error from 5.46% to 2.41%.

(retrieved 3 chunks in 0.41s)
  [1] docs/DILLI-GRID-Delhi-Electricity-Load-Forecasting.md
  [2] docs/DILLI-GRID-Delhi-Electricity-Load-Forecasting.md
  [3] docs/iex-price-forecasting.md
```

## Notes / next steps

- Swap `docs/*.md` for any corpus — internal wikis, contracts, support docs —
  the pipeline is corpus-agnostic.
- `k` (chunks retrieved) and `chunk_size`/`chunk_overlap` are tunable in
  `ingest.py` / `query.py` for retrieval-quality experiments.
- Could be extended with a re-ranking step or a Streamlit UI.
Diagnosed a retrieval-ranking failure where source project names weren't part of chunk text, causing relevant chunks (e.g. a results table) to rank below unrelated ones for named-entity queries. Fixed by prepending project context to each chunk before embedding.
