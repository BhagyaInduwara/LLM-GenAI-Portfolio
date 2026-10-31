# Week 5 — RAG Architecture

**Module:** PUSL3205 — Large Language Models and Generative AI
**Topic:** Retrieval-Augmented Generation (RAG) — building a document-grounded Q&A app

## Key concepts covered

- **RAG (Retrieval-Augmented Generation)** — instead of relying only on what an LLM learned during training, retrieve relevant chunks from your own documents and feed them to the LLM as context before it answers.
- **RAG workflow** — document loader → text splitter (chunking) → embeddings → vector store → retrieve similar chunks for a query → prompt template (query + retrieved context) → LLM → response.
- **LangChain** — an open-source framework for wiring LLMs together with external data sources, embedding models, vector databases, retrievers, and other tools.
- **Local model stack** — Llama 3.2 3B as the LLM and Qwen3 Embedding 0.6B as the embedding model, both served locally through Ollama (no API keys/cloud calls needed).
- **ChromaDB** — the vector database used to store and retrieve document embeddings.
- **Ingestion vs. querying as two separate stages** — `ingest.py` loads and embeds documents into the vector store once; `rag_pipeline.py` handles the actual question-answering loop against what's already stored.
- **Limitations of "naive" RAG** — fixed-size chunking, plain dense-vector top-k similarity search struggle with complex queries, long documents, exact terminology, and information spread across multiple sources (this is what the group activity below asks us to improve on).

## Resources (from lecturer)

- [Lab sheet 5.pdf](resources/Lab%20sheet%205.pdf) — Lab sheet: set up a local RAG stack (LangChain + Ollama + ChromaDB + Streamlit) and extend it; plus a group activity on advanced RAG architectures.

## What the Week 5 lab involves

**Individual part — build a basic RAG app:**
1. Set up a Python virtual environment (`rag_env`) and install LangChain, ChromaDB, Streamlit, PyMuPDF, python-docx, sentence-transformers, python-dotenv.
2. Pull a local LLM (`llama3.2:3b`) and an embedding model (`qwen3-embedding:0.6b`) via Ollama.
3. Build `ingest.py` to load a PDF (e.g. a university handbook), chunk it, embed it, and store it in ChromaDB.
4. Build `rag_pipeline.py`: take a user question, retrieve the most similar chunks, and generate an answer grounded in them.
5. Wrap it in a small Streamlit chat window.
6. **Task:** extend the app to read multiple PDFs, and let users upload their own PDF to ask questions against.

**Group activity — "From Naive RAG to Intelligent University Knowledge Retrieval":** pick (or get assigned) one limitation of the basic RAG system above and build an advanced architecture that addresses it. Five group options are listed in the lab sheet:

| Group | Focus |
|---|---|
| 1 | Hybrid retrieval (BM25 + vector) and reranking |
| 2 | Context-aware / hierarchical chunking (fixed-size vs. section-based vs. parent-child) |
| 3 | Corrective / evidence-verified RAG (retrieval-quality checker + retry) |
| 4 | Knowledge Graph RAG (entities, prerequisites, multi-hop questions) |
| 5 | Multilingual RAG (English, Sinhala, Tamil) |

Expected deliverables: a working enhanced RAG app, an architecture diagram, source code + setup instructions, an evaluation dataset (questions + reference answers + supporting documents), experimental results comparing baseline vs. the improved approach, a short technical report, and a live demo.

## Explaining my code: `lab/rag_app/`

The lab sheet's steps 3–6 are a real local app, not a notebook, so this is plain Python files instead of a `.ipynb`:

- **`rag_core.py`** — the shared logic every other file reuses:
  - `load_and_split_pdfs()` reads PDF(s) with `PyMuPDFLoader` and splits them into overlapping ~1000-character chunks with `RecursiveCharacterTextSplitter` (overlap keeps a sentence that straddles a chunk boundary from losing context).
  - `get_embeddings()` / `get_vectorstore()` wrap the Qwen3 embedding model and the Chroma vector store — this is the "embed → store" half of the RAG workflow from the slides.
  - `build_rag_chain()` wires together a retriever (top-k similarity search over Chroma), a prompt template that inserts the retrieved chunks as `{context}`, the local Llama 3.2 3B model via `ChatOllama`, and a parser that turns the model's reply into plain text — this is the "retrieve → prompt → LLM → response" half.
  - The prompt explicitly tells the model to answer **only** from the given context and say "I don't know" otherwise — this is what makes it RAG rather than just chatting with the base model.
- **`ingest.py`** — Lab sheet step "Build ingestion and run `ingest.py`": reads every PDF in a `documents/` folder, chunks and embeds them, and persists them into the Chroma store on disk.
- **`rag_pipeline.py`** — Lab sheet step "Build rag pipeline and run `rag_pipeline.py`": the terminal chat loop from the lab sheet's screenshot (`Ask a university question (or exit):`), answering from whatever `ingest.py` already stored.
- **`app.py`** — the lab sheet's **Task**: a Streamlit version (`streamlit run app.py`) where a user uploads one or more of their *own* PDFs through the browser, the app indexes them on the spot into a fresh in-memory Chroma store (so one user's uploads never leak into another's session), and then answers questions through the same chat chain — directly satisfying "read more pdfs" and "users can upload their own pdfs."
- **`.env.example`** — all the tunable settings (model names, chunk size, folder paths) read via `python-dotenv`, so nothing is hard-coded.

## How to run it locally

```
cd lab/rag_app
py -3.12 -m venv rag_env
.\rag_env\Scripts\Activate.ps1
pip install -r ../../../../requirements.txt
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
copy .env.example .env

mkdir documents          # drop one or more PDFs in here
python ingest.py         # builds the Chroma store from documents/
python rag_pipeline.py   # terminal chat, or:
streamlit run app.py     # browser chat with your own PDF uploads
```

This hasn't been run here (no local Ollama/GPU in this environment) — all four files passed a Python syntax check, but the actual RAG behaviour needs Ollama running locally with both models pulled.

## My work

- [notes.md](notes.md) — my notes
- [lab/rag_app/](lab/rag_app/) — my completed version of the individual lab: `rag_core.py`, `ingest.py`, `rag_pipeline.py`, and `app.py` (the multi-PDF/upload task). Not yet run — see "How to run it locally" above.
- [reflection.md](reflection.md) — my reflection
