# Week 1 Reflection

## What I learned

How classical NLP represents text as numbers before any modern embedding technique existed: one-hot encoding, then Bag-of-Words, then N-grams — and specifically *why* each one was replaced (no semantics, no word order, sparse vectors). That framing made Word2Vec and contextual embeddings make a lot more sense, since they're direct fixes for those three problems rather than unrelated new ideas.

## What was challenging

Keeping the different representation techniques straight, since several of them (one-hot, BoW, N-grams) look similar on paper but solve different problems. I also found it easy to conflate stemming and lemmatization until I worked through the "studies → study" vs "studies → studi" example by hand.

## How it connects to earlier weeks

This is the first content week, so there's no earlier week to connect it to yet — it's the baseline the rest of the module (attention, Transformers, LLMs) builds on.

## Questions I still have

Why some production systems still use TF-IDF/BoW instead of embeddings for simple tasks — is it purely a speed/cost trade-off, or are there tasks where sparse representations are genuinely more robust?

## What I'd try next

Run `lab/sentiment_analysis_traditional_ml.ipynb` end to end and compare Logistic Regression against an SVM on the same Bag-of-Words features, to see how much the choice of classifier actually matters once the representation is fixed.
