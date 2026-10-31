"""Lab Sheet 5 — "Build rag pipeline and run rag_pipeline.py"

Loads the Chroma vector store built by ingest.py and answers questions about
its contents from the terminal — the same "Ask a university question (or exit)"
loop shown in the lab sheet.

Usage:
    python rag_pipeline.py
"""
from rag_core import get_vectorstore, build_rag_chain, CHROMA_DIR


def main():
    vectorstore = get_vectorstore(CHROMA_DIR)
    chain = build_rag_chain(vectorstore)

    print("RAG pipeline ready. Run ingest.py first if you haven't already.")

    while True:
        question = input("\nAsk a university question (or exit): ").strip()

        if question.lower() == "exit":
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
