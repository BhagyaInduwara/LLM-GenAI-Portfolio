"""Group 5 — Multilingual University RAG: retrieval evaluation.

Runs every question in eval_questions.json, in English, Sinhala and Tamil,
through every embedding model in EMBEDDING_MODELS with two query strategies:

  direct      cross-lingual retrieval — the question is embedded as-is, in its
              own language, and matched against the English chunks
  translated  the question is first translated to English by the local LLM,
              then embedded (English questions are left unchanged)

A retrieved chunk counts as relevant if it contains one of the question's
answer keywords. Reports Hit@1, Hit@k and MRR per model / strategy / language,
and writes everything to results/.

Usage:
    python evaluate.py              # retrieval metrics only
    python evaluate.py --answers    # also generate answers with the best setup
"""
import argparse
import json
from pathlib import Path

from multilingual_core import (
    EMBEDDING_MODELS, LANGUAGES, TOP_K, Index, load_chunks, build_translator, answer,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
STRATEGIES = ["direct", "translated"]


def normalize(text):
    return " ".join(text.split()).lower()


def first_relevant_rank(ranking, chunks, keywords):
    """1-based rank of the first chunk containing an answer keyword (None if none)."""
    keys = [normalize(k) for k in keywords]
    for rank, i in enumerate(ranking, 1):
        if any(k in normalize(chunks[i].page_content) for k in keys):
            return rank
    return None


def metrics(ranks, k=TOP_K):
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks if r) / n,
        f"hit@{k}": sum(r <= k for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers with the best model/strategy")
    args = parser.parse_args()

    questions = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
    chunks = load_chunks()
    print(f"{len(questions)} questions x {len(LANGUAGES)} languages, {len(chunks)} chunks\n")

    print("Translating Sinhala and Tamil questions to English...")
    translate = build_translator()
    translations = {
        (q["id"], lang): translate(q[lang], lang) for q in questions for lang in LANGUAGES
    }

    results = []  # one row per (model, strategy, language)
    misses = []
    for model_name, make_embedder in EMBEDDING_MODELS.items():
        print(f"Indexing with {model_name}...")
        index = Index(chunks, make_embedder())
        for strategy in STRATEGIES:
            for lang in LANGUAGES:
                ranks = []
                for q in questions:
                    query = q[lang] if strategy == "direct" else translations[(q["id"], lang)]
                    rank = first_relevant_rank(index.rank(query), chunks, q["keywords"])
                    ranks.append(rank)
                    if rank is None or rank > TOP_K:
                        misses.append((model_name, strategy, lang, q["id"], query, rank))
                results.append({"model": model_name, "strategy": strategy,
                                "language": lang, **metrics(ranks), "ranks": ranks})

    RESULTS_DIR.mkdir(exist_ok=True)
    report = build_report(questions, results, misses, translations)
    (RESULTS_DIR / "results.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "results.json").write_text(json.dumps({
        "results": results,
        "translations": [{"id": i, "language": l, "translation": t}
                         for (i, l), t in translations.items() if l != "en"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"Saved to {RESULTS_DIR / 'results.md'}")

    if args.answers:
        generate_answers(questions, chunks, results, translations)


def build_report(questions, results, misses, translations):
    k = TOP_K
    lines = [
        f"## Retrieval results (Hit@1 / Hit@{k} / MRR)\n",
        "| Embedding model | Strategy | English | Sinhala | Tamil |",
        "|---|---|---|---|---|",
    ]
    for model_name in EMBEDDING_MODELS:
        for strategy in STRATEGIES:
            cells = []
            for lang in LANGUAGES:
                r = next(x for x in results if x["model"] == model_name
                         and x["strategy"] == strategy and x["language"] == lang)
                cells.append(f"{r['hit@1']:.2f} / {r[f'hit@{k}']:.2f} / {r['mrr']:.2f}")
            lines.append(f"| {model_name} | {strategy} | " + " | ".join(cells) + " |")

    lines += ["", "## LLM translations of the questions\n",
              "| # | Sinhala → English | Tamil → English |", "|---|---|---|"]
    for q in questions:
        lines.append(f"| {q['id']} | {translations[(q['id'], 'si')]} | "
                     f"{translations[(q['id'], 'ta')]} |")

    lines += ["", f"## Misses (answer chunk not in top {k})\n",
              "| Model | Strategy | Lang | Q# | Rank of answer chunk |", "|---|---|---|---|---|"]
    for model_name, strategy, lang, qid, _query, rank in misses:
        lines.append(f"| {model_name} | {strategy} | {lang} | {qid} | {rank or '-'} |")
    return "\n".join(lines) + "\n"


def generate_answers(questions, chunks, results, translations):
    """Answer every question with the best non-English setup, for manual review."""
    non_en = [r for r in results if r["language"] != "en"]
    best = max(
        {(r["model"], r["strategy"]) for r in non_en},
        key=lambda ms: sum(r["mrr"] for r in non_en if (r["model"], r["strategy"]) == ms),
    )
    model_name, strategy = best
    print(f"\nGenerating answers with {model_name} / {strategy}...")
    index = Index(chunks, EMBEDDING_MODELS[model_name]())

    lines = [f"## Generated answers ({model_name}, {strategy})\n"]
    for q in questions:
        lines.append(f"### Q{q['id']}. {q['en']}\n\n**Reference:** {q['reference_answer']}\n")
        for lang, name in LANGUAGES.items():
            query = q[lang] if strategy == "direct" else translations[(q["id"], lang)]
            reply = answer(q[lang], index.search(query), lang)
            lines.append(f"- **{name}** ({q[lang]}): {' '.join(reply.split())}")
        lines.append("")
    (RESULTS_DIR / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {RESULTS_DIR / 'answers.md'}")


if __name__ == "__main__":
    main()
