# Week 1 — NLP Foundations

**Module:** PUSL3205 — Large Language Models and Generative AI
**Topic:** From classical NLP to text representation and neural NLP

## Key concepts covered

- **What is NLP** — the field connecting human language and computers (search engines, voice assistants, translation).
- **Core NLP tasks** — sentiment analysis, named entity recognition (NER), text classification, speech recognition, chatbots/Q&A.
- **Text preprocessing** — lowercasing, removing punctuation/HTML/URLs, tokenization, stop-word removal, stemming vs lemmatization.
- **Text representation** — one-hot encoding, Bag-of-Words (BoW), N-grams, and why they're limited (no semantics, no word order, sparse vectors).
- **Word embeddings** — Word2Vec (CBOW/skip-gram) and contextual embeddings (BERT) that fix those limitations.
- **Neural architectures for text** — RNN, LSTM, CNN, and Transformers, compared side by side.

## Resources (from lecturer)

- [Week 1.pdf](resources/Week%201.pdf) — Lesson 1 slides: NLP, text preprocessing, text representation, neural NLP overview.
- [Lab sheet 1.pdf](resources/Lab%20sheet%201.pdf) — Lab sheet: sentiment analysis with traditional ML.

## What the Week 1 lab involves

Lab sheet 1 asks us to build a traditional (non-deep-learning) sentiment classifier:

1. Load the [Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences) (short product/movie/restaurant reviews labelled positive or negative).
2. Clean the text — lowercase it, strip punctuation and extra whitespace.
3. Explore the data — look at example reviews and the most frequent words in each class.
4. Split the data into train/test sets.
5. Convert the cleaned text into a **Bag-of-Words** vector (a table of word counts per review — this is exactly the BoW technique from the lecture slides).
6. Train a classic ML model (e.g. Logistic Regression or SVM) on those vectors.
7. Evaluate it with accuracy, precision, recall, and F1-score.
8. Try the trained model on a brand-new sentence.

No starter code was provided for this lab.

## My work

- [notes.md](notes.md) — my notes
- [lab/sentiment_analysis_traditional_ml.ipynb](lab/sentiment_analysis_traditional_ml.ipynb) — my completed version of the lab, covering all 8 steps from the lab sheet: downloading the UCI dataset, cleaning the text, exploring positive/negative word frequencies, a train/test split, a Bag-of-Words vectorizer, a Logistic Regression classifier, accuracy/precision/recall/F1 evaluation, and testing on new sentences. Not yet run — run it top to bottom to download the dataset and train the model.
- [reflection.md](reflection.md) — my reflection
