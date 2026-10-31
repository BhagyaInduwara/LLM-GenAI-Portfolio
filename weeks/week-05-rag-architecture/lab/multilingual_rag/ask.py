"""Group 5 — Multilingual University RAG: interactive chat.

Ask a question in English, Sinhala or Tamil. The language is detected from the
script, relevant chunks are retrieved, and the answer is written back in the
same language.

Usage:
    python ask.py                                   # cross-lingual (direct) retrieval
    python ask.py --translate                       # translate the question to English first
    python ask.py --model "multilingual-e5-small"   # pick an embedding model
"""
import argparse

from multilingual_core import (
    EMBEDDING_MODELS, LANGUAGES, Index, load_chunks, detect_language, build_translator, answer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-embedding:0.6b (Ollama)",
                        choices=list(EMBEDDING_MODELS))
    parser.add_argument("--translate", action="store_true",
                        help="translate non-English questions to English before retrieval")
    args = parser.parse_args()

    index = Index(load_chunks(), EMBEDDING_MODELS[args.model]())
    translate = build_translator() if args.translate else None
    print(f"Ready ({args.model}, {'translated' if args.translate else 'direct'} retrieval).")

    while True:
        question = input("\nAsk a university question in English / සිංහල / தமிழ் (or exit): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        lang = detect_language(question)
        query = translate(question, lang) if translate else question
        if query != question:
            print(f"  [{LANGUAGES[lang]} → English] {query}")

        chunks = index.search(query)
        print(f"  [sources] {', '.join(c.metadata['source'] for c in chunks)}")
        print(f"\nAnswer ({LANGUAGES[lang]}): {answer(question, chunks, lang)}")


if __name__ == "__main__":
    main()
