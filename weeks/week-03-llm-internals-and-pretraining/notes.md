# Week 3 Notes

## Key terms

- **Pipeline** — tokenizer → stack of Transformer blocks → LM head → probability distribution over the next token.
- **Autoregressive generation** — one token at a time; each new token gets appended to the prompt before the next forward pass.
- **Decoding** — greedy (always highest probability) vs. sampling (draw from the probability distribution — more varied, less deterministic).
- **Context length** — the maximum number of tokens a model can process at once; **caching** avoids recomputing earlier tokens on every new forward pass.
- **Encoder vs decoder** — encoders (BERT) read and represent text bidirectionally; decoders (GPT) generate text one token at a time, left to right only.
- **Pre-training objectives** — Masked Language Modeling (BERT: predict hidden tokens using both directions), Causal Language Modeling (GPT: predict the next token using only past tokens), Span Corruption (T5: reconstruct a removed span of text).
- **BERT vs GPT vs T5** — encoder-only / decoder-only / encoder-decoder; best for classification-QA / generation / translation-summarisation respectively.
- **Why decoder-only dominates in 2024+** — KV-caching is efficient, no cross-attention overhead, simpler to train at scale with next-token prediction.

## Important formulas / code

- Perplexity: `exp(validation_loss)` — lower is better, standard measure of how "surprised" a model is by real text.
- Causal mask: `torch.triu(torch.ones(T, T), diagonal=1)` — same mechanism as Week 2, applied inside a full LM this time.
- Parameters live in: embedding layer, attention layers, feed-forward layers, output (LM head) layer — more parameters ≠ automatically a better model; training data, architecture, and tokenizer all matter too.

## Open questions

- Why do decoder-only models dominate now even for tasks like translation, which T5's encoder-decoder design was originally built for?
- How much does character-level tokenization (used in the tiny Transformer lab) hurt a model's quality compared to sub-word tokenization, for the same parameter budget?
