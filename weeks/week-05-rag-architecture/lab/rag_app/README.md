# Week 5 RAG App — How to Run

A local RAG (Retrieval-Augmented Generation) app: LangChain + Ollama (Llama 3.2 3B, Qwen3 Embedding 0.6B) + ChromaDB + Streamlit. See the [week README](../../README.md) for what each file does.

> Run every command **from this folder** (`lab/rag_app`). The paths in `.env` (`documents`, `chroma_db`) are relative to it.

## Quick run (already set up)

If the packages are installed, both models are pulled, and `chroma_db/` already exists, you only need:

```powershell
cd "weeks/week-05-rag-architecture/lab/rag_app"

python rag_pipeline.py   # terminal chat over the PDFs in documents/
# or
python -m streamlit run app.py   # browser chat: upload your own PDFs
```

Ollama has to be running (the Ollama desktop app, or `ollama serve`). Type `exit` to quit the terminal chat. Streamlit opens at http://localhost:8501.

## When you change the PDFs in `documents/`

`ingest.py` **adds** chunks to the existing store. Running it twice on the same PDFs stores every chunk twice. To rebuild from scratch, delete the store first:

```powershell
Remove-Item -Recurse -Force chroma_db
python ingest.py
```

`app.py` doesn't use `documents/` or `chroma_db/`. It indexes whatever you upload in the browser, in memory, for that session only.

## Reranking

Retrieval has two steps: vector search fetches `FETCH_K` (20) candidate chunks, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, about 90 MB, downloaded from Hugging Face on first use) re-scores each chunk against the question and keeps the best `TOP_K` (4). It's on by default in both `rag_pipeline.py` and `app.py`.

To see what reranking changes for a question (retrieval only, no LLM call):

```powershell
python compare_rerank.py "Can I use ChatGPT in my assignments?"
```

To switch it off and get the original behaviour, set `RERANK=false` in `.env`.

### Before vs after results

Tested on 2026-10-31 against the 5 PDFs in `documents/` (12 chunks in total), with `TOP_K=4`, `FETCH_K=20`, and Llama 3.2 3B at temperature 0. **Before** = vector search top 4. **After** = vector search top 20, reranked, keep top 4.

| # | Question | Top-4 chunks before | Top-4 chunks after (rerank score of #1) | Answer |
|---|---|---|---|---|
| 1 | Can I use ChatGPT in my assignments? | 3× GenAI Policy, 1× Capstone | 2× GenAI Policy, Capstone, Exam Regs (3.05) | Same, correct |
| 2 | What is the penalty for not declaring GenAI usage? | 3× GenAI Policy, 1× Capstone | 3× GenAI Policy, Library (8.67) | Same, correct |
| 3 | What attendance is required to sit an exam? | 2× Exam Regs, Capstone, Library | 2× Exam Regs, 2× BSc Syllabus (6.11) | **Better after**: adds "in lectures, practical sessions, and tutorials" |
| 4 | How long can I borrow a book from the library? | 2× Library, Capstone, GenAI Policy | 2× Library, BSc Syllabus, Exam Regs (4.15) | Same, correct (5 books, 14 days) |
| 5 | What is the deadline to submit an academic appeal? | **1×** Exam Regs, Capstone, GenAI Policy, Library | **2×** Exam Regs, Capstone, GenAI Policy (2.23) | Same, correct (14 working days) |
| 6 | What topics can I choose for my capstone project? | 1× Capstone, BSc Syllabus, GenAI Policy, Library | **2× Capstone**, Library, GenAI Policy (3.69) | **Better after**: lists all 3 research areas instead of 1 |

**Question 6 is the clearest example.** The capstone topic list is split across two chunks. Vector search only found the first one, so the model listed just one topic (Advanced RAG). The reranker pulled the second chunk up to #2, and the answer then listed all three: Advanced RAG, Autonomous AI Agents, and Edge AI & Quantization.

**What the results show:**
- **Ranking:** the most relevant chunk was #1 in both versions every time. Reranking mostly changes positions 2 to 4.
- **Confidence:** the reranker clearly separates the right chunk from the rest. The #1 chunk scored between +2 and +9, while unrelated chunks scored around −10. Vector search gives no such signal.
- **Fewer off-topic chunks:** the Capstone "Advanced RAG" chunk showed up in the baseline top 4 for unrelated questions (1, 2 and 3) because it shares AI vocabulary with them. After reranking, it dropped out of question 3's top 4.
- **Reranking is not always better:** for question 1, it pushed the "Penalties for Non-Compliance" chunk out of the top 4, even though that chunk is relevant to "can I use ChatGPT".
- **The test set is small:** with only 12 chunks, `FETCH_K=20` means the reranker re-orders the whole store. The gap should get bigger with more documents, where vector search returns more near-misses.

## First-time setup (new machine only)

Skip this section if the quick run above works.

1. **Install Ollama** from https://ollama.com, then pull the two models:
   ```powershell
   ollama pull llama3.2:3b
   ollama pull qwen3-embedding:0.6b
   ```
   Check with `ollama list`.

2. **Install only the packages this app needs.** You don't need the repo-wide `requirements.txt`, which also installs torch, transformers and other packages from earlier weeks:
   ```powershell
   pip install langchain langchain-community langchain-ollama langchain-chroma langchain-text-splitters chromadb streamlit pymupdf python-dotenv sentence-transformers
   ```
   `sentence-transformers` (which brings in torch) is only needed for reranking. Without it, set `RERANK=false` in `.env`.
   A virtual environment is optional. If you want one: `python -m venv rag_env` then `.\rag_env\Scripts\Activate.ps1` before the `pip install`.

3. **Create `.env`** from the template (model names, chunk size, top-k):
   ```powershell
   copy .env.example .env
   ```

4. **Add PDFs to `documents/` and build the vector store:**
   ```powershell
   python ingest.py
   ```

Then go back to [Quick run](#quick-run-already-set-up).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` / `httpx.ConnectError` | Ollama isn't running. Start the Ollama app or run `ollama serve`. |
| `model "..." not found` | Pull it with `ollama pull <model>`, or fix the name in `.env`. |
| Answers are always "I don't know" | `chroma_db/` is empty or missing. Run `python ingest.py` from this folder. |
| Same source repeated in answers | Duplicate chunks from running `ingest.py` more than once. Delete `chroma_db/` and re-ingest. |
| `'streamlit' is not recognized as an internal or external command` | Python's `Scripts` folder isn't on PATH. Use `python -m streamlit run app.py` instead. |
| `ModuleNotFoundError` | Run the `pip install` line from step 2, in the same Python/venv you run the app with. |
