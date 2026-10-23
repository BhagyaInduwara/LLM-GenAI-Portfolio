# Week 3 — LLM Internals and Pre-training

**Module:** PUSL3205 — Large Language Models and Generative AI
**Topic:** How a Transformer LLM generates text, and how LLMs are pre-trained

## Key concepts covered

- **Input → output pipeline** — tokenizer → stack of Transformer blocks → LM head → next-token probabilities.
- **Autoregressive generation** — the model produces **one token at a time**, each time appending its own output back onto the prompt before the next forward pass.
- **Decoding/sampling** — greedy decoding (always pick the highest-probability token) vs. random sampling.
- **Context length & caching** — the limit on how many tokens a model can process at once, and why caching previous computations speeds up generation.
- **Encoder vs decoder** — encoders (BERT) read and represent text; decoders (GPT) generate text one token at a time.
- **Pre-training objectives** — Masked Language Modeling (BERT), Causal Language Modeling (GPT), and Span Corruption (T5).
- **BERT vs GPT vs T5** — architecture, masking style, and best-fit tasks for each.
- **The modern LLM family** — GPT-4, LLaMA-3, Mistral, Gemma, Qwen, and why decoder-only architectures dominate in 2024–25 (efficient KV-caching, simpler training).
- **Parameter counts** — where a model's parameters live (embeddings, attention, feed-forward, output layers) and why more parameters ≠ automatically a better model.

## Resources (from lecturer)

- [Week 3.pdf](resources/Week%203.pdf) — Lesson 3 slides: looking inside an LLM, pre-training objectives, BERT vs GPT vs T5, the modern LLM family.
- [Lab sheet 3.pdf](resources/Lab%20sheet%203.pdf) — Lab sheet: build, train and evaluate a tiny Transformer language model from scratch.
- [tiny_transformer_llm.ipynb](resources/tiny_transformer_llm.ipynb) — Lecturer-provided reference notebook for the lab.

## What the Week 3 lab involves

Lab sheet 3 walks through building a small character-level Transformer language model end to end, using the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) (short, simple children's stories): load and explore the data, tokenize it, build the Transformer architecture, train it, track the loss curve, evaluate with perplexity, generate new text from a prompt, experiment with model settings (layers/heads/epochs), save the trained model, and optionally wrap it in a Gradio chat window.

## Explaining the lecturer's code: `tiny_transformer_llm.ipynb`

This is a complete, working mini-LLM. Step by step:

1. **Load data** — downloads 1,000 short stories from the `roneneldan/TinyStories` dataset on Hugging Face and splits them into train/validation sets.
2. **Character-level tokenizer** — instead of splitting text into words or sub-words, every unique character in the dataset becomes one token (`stoi`/`itos` = "string to index" / "index to string"). Simpler than a real LLM's tokenizer, but the same idea.
3. **`TinyTransformer` model**:
   - `token_emb` + `pos_emb` — the same embedding + positional-encoding idea as Week 2, now using PyTorch's ready-made `nn.TransformerEncoderLayer` instead of a hand-built one.
   - `nn.TransformerEncoder` with 2 layers, 2 attention heads — this **is** the "stack of Transformer blocks" from the Week 3 slides.
   - A **causal mask** (`torch.triu(..., diagonal=1)`) is passed into the encoder so each position can only attend to earlier positions — again, what turns an encoder into a GPT-style autoregressive language model.
   - `lm_head` — a final linear layer that turns the model's internal representation into a probability distribution over the vocabulary, exactly the "LM head" described in the slides.
4. **Batching (`get_batch`)** — randomly samples fixed-length chunks of text; the target sequence `y` is just the input sequence `x` shifted one character to the right (classic next-token-prediction setup).
5. **Training loop** — for 15 epochs, runs 100 batches each: forward pass → cross-entropy loss → `loss.backward()` (backpropagation) → `optimizer.step()` (AdamW updates the weights). Training and validation loss are recorded every epoch.
6. **Loss curve** — plots training vs. validation loss with Matplotlib to check the model is actually learning (and not overfitting).
7. **Perplexity** — `exp(validation_loss)`, a standard metric for how "surprised" the model is by real text (lower = better).
8. **Saving** — stores the trained weights plus the vocabulary and config in a checkpoint file (`tiny_transformer.pt`) so the model can be reloaded later without retraining.
9. **Gradio chat demo** — wraps a `generate()` function in a simple chat UI so the tiny model can be talked to directly in the browser. **Bug in this notebook:** it checks `if "generate" not in globals()` and calls `generate(...)`, but `generate()` is never actually defined anywhere in the notebook — Lab sheet 3's Step 9 ("generate stories from a prompt") is missing entirely, so the chat demo would fail as-is. Fixed in [My work](#my-work) below.

## My work

- [notes.md](notes.md) — my notes
- [lab/tiny_transformer_llm_complete.ipynb](lab/tiny_transformer_llm_complete.ipynb) — my completed version of the lab. Builds on the lecturer's notebook above, keeping Steps 1–8 as they were, and adds the two things it was missing: a working `generate()` function for Step 9 (same one-character-at-a-time sampling idea as Week 2's `generate`, so the Gradio chat cell actually works), and a Step 10 cell experimenting with model settings (`TinyTransformer` now takes `num_layers`/`nhead`/`dim_feedforward` as arguments instead of having them hard-coded, so different sizes can be compared). Not yet run.
- [reflection.md](reflection.md) — my reflection
