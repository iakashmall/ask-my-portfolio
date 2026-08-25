"""
query.py — ask questions about your own projects, answered via RAG.

Uses a free, local LLM via Ollama — no API key, no cost.

Usage:
    python query.py "What model did the load forecasting project use?"
"""

import sys
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"  # pull this once: `ollama pull llama3.2`

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def build_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return retriever, llm


def ask(retriever, llm, question: str):
    start = time.time()

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    messages = PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    latency = time.time() - start

    print(f"\nQ: {question}")
    print(f"A: {response.content}")
    print(f"\n(retrieved {len(docs)} chunks in {latency:.2f}s)")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {src}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What projects use time-series forecasting?"
    retriever, llm = build_pipeline()
    ask(retriever, llm, question)