# Week 1 Notes

## Key terms

- **NLP** — field connecting human language and computers; tasks include sentiment analysis, NER, classification, speech recognition, chatbots/Q&A.
- **Preprocessing pipeline** — lowercasing → remove punctuation/HTML/URLs → tokenization → stop-word removal → stemming or lemmatization.
- **Tokenization levels** — sentence, word, character, sub-word (BERT/GPT/T5 use sub-word to balance vocabulary size against out-of-vocabulary words).
- **Stemming vs lemmatization** — stemming is fast and rule-based but can produce non-words (`happily → happi`); lemmatization always returns a real dictionary word (`studies → study`) but is slower.
- **Text representation, in order of sophistication** — one-hot encoding → Bag-of-Words (BoW) → N-grams → Word2Vec → contextual embeddings (BERT). Each step fixes a limitation of the one before it.
- **Neural architectures for sequences** — RNN (sequential, no long-range memory) → LSTM (gated memory) → CNN (local pattern detector) → Transformer (parallel, self-attention, long-range by default).

## Important formulas / code

- One-hot: vocabulary `[cat, dog, bird]` → `cat = [1,0,0]`.
- BoW: each document becomes a row of word counts over the shared vocabulary (`CountVectorizer` in scikit-learn).
- Word2Vec semantic arithmetic: `King − Man + Woman ≈ Queen`.
- N-grams preserve local order that BoW throws away: bigrams of "machine learning" → `[machine learning]` as one unit.

## Open questions

- At roughly what dataset size does BoW/TF-IDF stop being competitive with embeddings for a simple binary classification task like sentiment?
- How much does sub-word tokenization actually reduce the out-of-vocabulary problem in practice, versus just picking a bigger word-level vocabulary?
