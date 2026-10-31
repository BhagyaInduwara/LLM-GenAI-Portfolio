"""Shared building blocks used by ingest.py, rag_pipeline.py, and app.py:
loading/chunking PDFs, the embedding model, the vector store, and the
retrieve -> prompt -> LLM chain that actually answers a question.
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
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

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


def _format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs
    )


def build_rag_chain(vectorstore, k=TOP_K):
    """Build a retrieve -> prompt -> LLM -> text chain for question answering."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
