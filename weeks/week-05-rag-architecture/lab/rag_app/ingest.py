"""Lab Sheet 5 — "Build ingestion and run ingest.py"

Loads every PDF in DOCS_DIR, splits it into chunks, embeds the chunks with the
local Qwen3 embedding model (via Ollama), and persists them into a Chroma
vector store so rag_pipeline.py / app.py can retrieve from it later.

Usage:
    python ingest.py
"""
import os
from pathlib import Path

from rag_core import load_and_split_pdfs, get_embeddings, get_vectorstore, CHROMA_DIR

DOCS_DIR = Path(os.getenv("DOCS_DIR", "documents"))


def main():
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {DOCS_DIR}/. Add one or more PDF files there and re-run.")
        return

    print(f"Found {len(pdf_paths)} PDF(s): {[p.name for p in pdf_paths]}")

    chunks = load_and_split_pdfs(pdf_paths)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(CHROMA_DIR, embeddings)
    vectorstore.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks from {len(pdf_paths)} PDF(s) into '{CHROMA_DIR}'")


if __name__ == "__main__":
    main()
