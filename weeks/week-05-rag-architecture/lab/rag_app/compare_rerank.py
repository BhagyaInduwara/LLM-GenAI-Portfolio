"""Show which chunks the retriever returns for a question, with and without
cross-encoder reranking, so the effect of reranking is visible side by side.

Uses the Chroma store built by ingest.py. No LLM call — retrieval only.

Usage:
    python compare_rerank.py "Can I use ChatGPT in my assignments?"
    python compare_rerank.py          # prompts for a question
"""
import sys

from rag_core import get_vectorstore, build_retriever, CHROMA_DIR, TOP_K, FETCH_K


def show(title, docs):
    print(f"\n=== {title} ===")
    for i, d in enumerate(docs, 1):
        score = d.metadata.get("rerank_score")
        score_text = f"  (rerank score {score:.2f})" if score is not None else ""
        snippet = " ".join(d.page_content.split())[:160]
        print(f"{i}. [{d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]{score_text}")
        print(f"   {snippet}...")


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ").strip()
    vectorstore = get_vectorstore(CHROMA_DIR)

    baseline = build_retriever(vectorstore, rerank=False).invoke(question)
    reranked = build_retriever(vectorstore, rerank=True).invoke(question)

    show(f"Vector search only (top {TOP_K})", baseline)
    show(f"Vector search top {FETCH_K} -> cross-encoder rerank -> top {TOP_K}", reranked)


if __name__ == "__main__":
    main()
