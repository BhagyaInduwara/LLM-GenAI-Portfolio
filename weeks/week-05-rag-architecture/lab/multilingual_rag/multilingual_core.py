"""Shared building blocks for the multilingual RAG experiments: the embedding
models being compared, query translation, language detection, retrieval,
and the answer chain.

Chunks come from ../rag_app (same PDFs, same chunking), so results here are
directly comparable with the English-only app.
"""
import os
import sys
from pathlib import Path

import numpy as np

RAG_APP_DIR = Path(__file__).resolve().parent.parent / "rag_app"
sys.path.insert(0, str(RAG_APP_DIR))

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402

from rag_core import load_and_split_pdfs, OLLAMA_LLM_MODEL, TOP_K  # noqa: E402

DOCS_DIR = Path(os.getenv("DOCS_DIR", RAG_APP_DIR / "documents"))
if not DOCS_DIR.is_absolute():
    DOCS_DIR = RAG_APP_DIR / DOCS_DIR

LANGUAGES = {"en": "English", "si": "Sinhala", "ta": "Tamil"}


class SentenceTransformerEmbedder:
    """A Hugging Face embedding model run locally via sentence-transformers."""

    def __init__(self, model_name, query_prefix="", passage_prefix=""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts):
        return self.model.encode(
            [self.passage_prefix + t for t in texts], normalize_embeddings=True
        )

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)


class OllamaEmbedder:
    """An embedding model served by the local Ollama install (as in rag_app)."""

    def __init__(self, model_name, query_prefix=""):
        self.query_prefix = query_prefix
        self._embeddings = OllamaEmbeddings(model=model_name)

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    def embed_documents(self, texts):
        return self._normalize(self._embeddings.embed_documents(texts))

    def embed_query(self, text):
        return self._normalize(self._embeddings.embed_query(self.query_prefix + text))


# The embedding models under comparison. Each needs its own index, because
# vectors from different models live in different spaces.
EMBEDDING_MODELS = {
    # English-only baseline: shows what happens without multilingual training.
    "all-MiniLM-L6-v2 (English-only)": lambda: SentenceTransformerEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    # Multilingual (~100 languages incl. Sinhala & Tamil). E5 models are
    # trained with these "query: " / "passage: " prefixes and expect them.
    "multilingual-e5-small": lambda: SentenceTransformerEmbedder(
        "intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    # The model rag_app already uses (multilingual). Qwen3 embeddings expect
    # an instruction on the query side only.
    "qwen3-embedding:0.6b (Ollama)": lambda: OllamaEmbedder(
        "qwen3-embedding:0.6b",
        query_prefix="Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
    ),
    # Stronger but a 2.2 GB download — uncomment to include it:
    # "bge-m3": lambda: SentenceTransformerEmbedder("BAAI/bge-m3"),
}


class Index:
    """A tiny in-memory vector index: chunk embeddings + cosine-similarity search.

    With only a dozen chunks, plain numpy is enough and keeps the comparison
    transparent — every model is searched exactly the same way.
    """

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.embed_documents([c.page_content for c in chunks])

    def rank(self, query):
        """Return all chunk indices, most similar first."""
        scores = self.matrix @ self.embedder.embed_query(query)
        return list(np.argsort(-scores))

    def search(self, query, k=TOP_K):
        return [self.chunks[i] for i in self.rank(query)[:k]]


def load_chunks():
    return load_and_split_pdfs(sorted(DOCS_DIR.glob("*.pdf")))


def detect_language(text):
    """Detect Sinhala/Tamil by Unicode script block; anything else is English."""
    sinhala = sum("඀" <= ch <= "෿" for ch in text)
    tamil = sum("஀" <= ch <= "௿" for ch in text)
    if sinhala == tamil == 0:
        return "en"
    return "si" if sinhala >= tamil else "ta"


TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    """Translate the following {language} question into English.
Reply with only the English translation, nothing else.

{question}"""
)


def build_translator():
    """Return a function that translates a (lang, question) pair into English."""
    chain = (
        TRANSLATE_PROMPT
        | ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)
        | StrOutputParser()
    )

    def translate(question, lang):
        if lang == "en":
            return question
        return chain.invoke(
            {"language": LANGUAGES[lang], "question": question}
        ).strip()

    return translate


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a university assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make something up.
Write your answer in {language}.

Context:
{context}

Question: {question}

Answer:"""
)


def answer(question, chunks, lang):
    """Generate an answer in the question's language, grounded in the given chunks."""
    context = "\n\n".join(
        f"[{c.metadata.get('source', 'unknown')}] {c.page_content}" for c in chunks
    )
    chain = (
        ANSWER_PROMPT
        | ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)
        | StrOutputParser()
    )
    return chain.invoke(
        {"context": context, "question": question, "language": LANGUAGES[lang]}
    )
