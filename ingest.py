"""
ingest.py — builds a local FAISS vector index over your project docs.

Run this once (or whenever docs/ changes):
    python ingest.py
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def add_source_context(chunks):
    """
    Prepend the project/file name to each chunk's text before embedding.

    Why: a chunk's raw text (e.g. a MAPE results table) often doesn't
    repeat the project name it belongs to. Small embedding models rely on
    literal word overlap more than large ones, so a query like
    "What MAPE did DILLI-GRID achieve?" can fail to match a chunk that never
    says "DILLI-GRID" even if the numbers are right there. Prepending the
    source name fixes this cheaply, without changing chunking or model size.
    """
    for c in chunks:
        source = c.metadata.get("source", "")
        project_name = (
            os.path.basename(source).replace(".md", "").replace("-", " ")
        )
        c.page_content = f"[Project: {project_name}]\n{c.page_content}"
    return chunks


def main():
    print(f"Loading documents from ./{DOCS_DIR} ...")
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    print(f"  Loaded {len(docs)} documents.")

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    chunks = add_source_context(chunks)
    print(f"  Produced {len(chunks)} chunks.")

    print(f"Embedding with {EMBED_MODEL} (runs locally, no API cost)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to ./{INDEX_DIR}")


if __name__ == "__main__":
    main()