# Multilingual University RAG (Group 5)

**Task:** Evaluate English, Sinhala, and Tamil questions against university documents. Compare multilingual embeddings, translated queries, and cross-lingual retrieval.

This builds on [`../rag_app`](../rag_app/). It uses the same 5 PDFs, chunked the same way (12 chunks), so results are directly comparable with the English-only app.

## What's compared

**Three embedding models:**

| Model | Size | Languages | Why it's included |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | English only | Baseline: what happens without multilingual training |
| `multilingual-e5-small` | ~470 MB | ~100, incl. Sinhala & Tamil | A small dedicated multilingual model |
| `qwen3-embedding:0.6b` (Ollama) | ~640 MB | 100+ | The model `rag_app` already uses |

**Two query strategies for each model:**

- **direct (cross-lingual retrieval):** the Sinhala/Tamil question is embedded as-is and matched straight against the English chunks. This only works if the embedding model maps different languages into a shared vector space.
- **translated (translated queries):** the local LLM (Llama 3.2 3B) first translates the question into English, then it's embedded. English questions are left unchanged, so their two rows are identical.

## Evaluation dataset

[`eval_questions.json`](eval_questions.json) has 12 questions covering all 5 PDFs. Each question has:
- the same question in English (`en`), Sinhala (`si`) and Tamil (`ta`)
- the source PDF
- a reference answer
- answer keywords: a retrieved chunk counts as relevant only if it contains one of them

> The Sinhala and Tamil questions were machine-written. **Check them with a native speaker** before using the results in a report. Product names (ChatGPT) and "AI" are kept in English, as students would actually type them.

**Metrics:**
- **Hit@1:** the answer chunk is ranked first.
- **Hit@4:** the answer chunk is in the top 4. That's what the LLM sees (`TOP_K=4`).
- **MRR (Mean Reciprocal Rank):** 1/rank of the answer chunk, averaged over the questions. 1.00 is perfect; random ranking over 12 chunks gives about 0.26.

## Results

Run on 2026-10-31. The full output, including every miss, is in [`results/results.md`](results/results.md).

**Hit@1 / Hit@4 / MRR**

| Embedding model | Strategy | English | Sinhala | Tamil |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (English-only) | direct | 0.83 / 0.92 / 0.87 | 0.25 / 0.50 / 0.40 | 0.33 / 0.42 / 0.44 |
| all-MiniLM-L6-v2 (English-only) | translated | 0.83 / 0.92 / 0.87 | 0.17 / 0.75 / 0.43 | 0.75 / 0.92 / 0.82 |
| multilingual-e5-small | direct | **0.92 / 0.92 / 0.93** | **0.67 / 0.92 / 0.79** | 0.67 / 0.83 / 0.78 |
| multilingual-e5-small | translated | **0.92 / 0.92 / 0.93** | 0.17 / 0.58 / 0.36 | **0.92 / 0.92 / 0.93** |
| qwen3-embedding:0.6b | direct | 0.83 / 0.92 / 0.89 | 0.17 / 0.75 / 0.39 | 0.25 / 0.67 / 0.51 |
| qwen3-embedding:0.6b | translated | 0.83 / 0.92 / 0.89 | 0.17 / 0.58 / 0.39 | 0.83 / 0.92 / 0.89 |

**Best setup per language:**

| Language | Best setup | MRR |
|---|---|---|
| English | multilingual-e5-small (either strategy) | 0.93 |
| Sinhala | multilingual-e5-small, **direct** | 0.79 |
| Tamil | multilingual-e5-small, **translated** | 0.93 |

### Findings

1. **An English-only embedding model fails on Sinhala and Tamil.** With direct queries, MiniLM drops from 0.87 MRR in English to 0.40 (Sinhala) and 0.44 (Tamil), not much better than random. What little it gets right comes from the English words inside the questions ("AI", "ChatGPT").

2. **A multilingual embedding model makes cross-lingual retrieval work without any translation.** `multilingual-e5-small` with direct queries reaches 0.79 (Sinhala) and 0.78 (Tamil) MRR, and also gives the best English score.

3. **"Multilingual" on the label isn't enough.** `qwen3-embedding:0.6b` claims 100+ languages, but with direct queries it only manages 0.39 (Sinhala) and 0.51 (Tamil). It puts the right chunk in the top 4 fairly often but rarely at #1. Language coverage has to be tested, not assumed.

4. **Translated queries depend entirely on how well the LLM translates, and Llama 3.2 3B can't translate Sinhala.** Its Tamil translations are accurate ("How many credits will be awarded for the final year research project?"). Its Sinhala translations are made up and unrelated to the question:

   | Q# | Actual question | Llama 3.2's "translation" of the Sinhala |
   |---|---|---|
   | 2 | What is the deadline to submit an academic appeal? | What is the earliest date of the death of Ashoka? |
   | 6 | How many credits is the final year capstone project worth? | How many types of lachanakiri are there in the Viharaya? |
   | 10 | Which items are prohibited in the examination hall? | Do you have a coconut tree in your village? |

   So translation is the **best** strategy for Tamil (0.89–0.93 MRR with every model, even the English-only one) and the **worst** for Sinhala (0.36–0.43). Llama 3.2 officially supports only 8 languages, and neither Sinhala nor Tamil is one of them. It has clearly seen far more Tamil than Sinhala.

5. **Question 7 (capstone research topics) is missed in every language by every model.** The list of topics is split across two chunks. Retrieval finds the chunk with the heading ("Key Research Domains") but not the one with the full list. This is a chunking problem, not a language problem (see Group 2), and it's the same issue reranking fixed in [`../rag_app`](../rag_app/README.md#before-vs-after-results).

### Recommendation

Route each query by language, since language is easy to detect from the script:
- **Sinhala:** direct cross-lingual retrieval with `multilingual-e5-small`. Don't translate with Llama 3.2.
- **Tamil and English:** translate to English (Tamil only), then retrieve with `multilingual-e5-small`.

A better translator would probably make translation work for Sinhala too. Options: a larger LLM, or a dedicated translation model such as NLLB-200, which supports both Sinhala and Tamil.

### Answer generation

<!-- ANSWERS -->

## How to run

Run these from this folder (`lab/multilingual_rag`). Ollama must be running with `llama3.2:3b` and `qwen3-embedding:0.6b` pulled, as for `rag_app`. No extra packages are needed beyond `rag_app`'s plus `sentence-transformers`, which is already installed for reranking. The two Hugging Face models download automatically on the first run.

```powershell
cd "weeks/week-05-rag-architecture/lab/multilingual_rag"

python evaluate.py             # retrieval comparison → results/results.md (a few minutes)
python evaluate.py --answers   # also generate answers in all 3 languages → results/answers.md (slow on CPU)

python ask.py                                          # chat in English / Sinhala / Tamil
python ask.py --model "multilingual-e5-small"          # pick an embedding model
python ask.py --model "multilingual-e5-small" --translate   # translate to English before retrieval
```

To add a model to the comparison (for example `BAAI/bge-m3`, a 2.2 GB download), uncomment or add it in `EMBEDDING_MODELS` in [`multilingual_core.py`](multilingual_core.py).

## Files

| File | What it does |
|---|---|
| [`eval_questions.json`](eval_questions.json) | 12 questions in EN/SI/TA with source PDF, reference answer, and answer keywords |
| [`multilingual_core.py`](multilingual_core.py) | Embedding models under comparison, in-memory vector index, language detection (by Unicode script), LLM query translation, and the answer prompt |
| [`evaluate.py`](evaluate.py) | Runs every question × language × model × strategy and computes Hit@1 / Hit@4 / MRR |
| [`ask.py`](ask.py) | Interactive chat: detects the question's language, retrieves, and answers in that language |
| [`results/`](results/) | Output of the last `evaluate.py` run |
