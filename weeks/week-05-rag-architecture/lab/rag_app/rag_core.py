"""Shared building blocks used by ingest.py, rag_pipeline.py, and app.py:
loading/chunking PDFs, the embedding model, the vector store, the optional
cross-encoder reranker, and the retrieve -> (rerank) -> prompt -> LLM chain
that actually answers a question.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Reranking: fetch FETCH_K candidates by vector similarity, then let a
# cross-encoder re-score them against the question and keep the best TOP_K.
RERANK = os.getenv("RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FETCH_K = int(os.getenv("FETCH_K", "20"))

PROMPT = ChatPromptTemplate.from_template(
    """You are a university assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make something up.

Context:
{context}

Question: {question}

Answer:"""
)


def load_and_split_pdfs(pdf_paths):
    """Load one or more PDF files and split them into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for path in pdf_paths:
        pages = PyMuPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["source"] = Path(path).name
        chunks.extend(splitter.split_documents(pages))
    return chunks


def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def get_vectorstore(persist_dir=CHROMA_DIR, embeddings=None):
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings or get_embeddings(),
    )


_reranker = None


def get_reranker():
    """Load the cross-encoder once (downloaded from Hugging Face on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank_docs(question, docs, top_n=TOP_K):
    """Score each (question, chunk) pair with the cross-encoder and keep the best top_n.

    Unlike embeddings, which encode the question and the chunk separately, a
    cross-encoder reads them together, so it judges relevance far more
    accurately — but it's too slow to run over every chunk, hence only on
    the FETCH_K candidates the vector search already shortlisted.
    """
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    for doc, score in ranked:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in ranked[:top_n]]


def build_retriever(vectorstore, k=TOP_K, rerank=RERANK, fetch_k=FETCH_K):
    """Return a runnable mapping a question to its k most relevant chunks."""
    if not rerank:
        return vectorstore.as_retriever(search_kwargs={"k": k})

    candidates = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    return RunnableLambda(lambda q: rerank_docs(q, candidates.invoke(q), k))


def _format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs
    )


def build_rag_chain(vectorstore, k=TOP_K, rerank=RERANK):
    """Build a retrieve -> (rerank) -> prompt -> LLM -> text chain for question answering."""
    retriever = build_retriever(vectorstore, k=k, rerank=rerank)
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
