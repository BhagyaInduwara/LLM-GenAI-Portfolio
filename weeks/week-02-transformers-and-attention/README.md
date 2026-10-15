# Week 2 — Transformers and Attention

**Module:** PUSL3205 — Large Language Models and Generative AI
**Topic:** Neural network foundations, self-attention, and the Transformer architecture

## Key concepts covered

- **The neuron** — `output = activation(Σ wᵢxᵢ + b)`, and why activation functions matter (ReLU, GELU, SwiGLU used in modern LLMs).
- **MLP forward pass & backpropagation** — how a multi-layer network computes a prediction and updates its weights via the chain rule.
- **Vanishing gradients** — why deep sigmoid/tanh networks struggle to train, and how residual connections + LayerNorm + GELU fix it.
- **Why RNNs weren't enough** — sequential processing, no long-range memory, can't parallelise.
- **Query, Key, Value (Q/K/V)** — the three roles every token plays in self-attention.
- **Scaled dot-product attention & multi-head attention** — computing attention scores, softmax, and running several attention "heads" in parallel to capture different relationships (syntax, coreference, long-range dependencies).
- **"Attention Is All You Need" (Vaswani et al., 2017)** — the paper that introduced the Transformer, replacing recurrence with self-attention entirely.

## Resources (from lecturer)

- [Week 2.pdf](resources/Week%202.pdf) — Lesson 2 slides: neural network basics, attention mechanism, multi-head attention, and an "AI in everyday challenges" product design exercise.
- [Lab sheet 2.pdf](resources/Lab%20sheet%202.pdf) — Group challenge cards: use an LLM as a product-development partner to design a solution to an everyday student problem.
- [1706.03762v7.pdf](resources/1706.03762v7.pdf) — The original "Attention Is All You Need" paper.
- [transformer_practice.ipynb](resources/transformer_practice.ipynb) — Lecturer-provided example notebook: a minimal Transformer block built by hand.

## What the Week 2 lab involves

Lab sheet 2 ("AI in everyday challenges") is a group design exercise, not a coding task: each group gets a real student problem (forgetfulness, distraction, food waste, dehydration, time management, plastic waste, overloaded bags, or budgeting) and uses an LLM as a brainstorming partner to design a product concept, name, key features, target user, and a mock-up, then presents it in 2–3 minutes.

## Explaining the lecturer's code: `transformer_practice.ipynb`

This script builds a tiny Transformer **by hand** (rather than using a ready-made `nn.TransformerEncoderLayer`), to show what's actually happening inside self-attention. In plain terms:

1. **Toy dataset** — a few repeated sentences about a cat and a dog (`"the cat sat on the mat."` etc.), tokenized **character by character** (each letter/space/punctuation mark is one token).
2. **Embeddings** — every character gets a learned vector (`token_embedding`), and its position in the sequence gets another learned vector (`position_embedding`). These are added together so the model knows both *what* each character is and *where* it sits.
3. **Self-attention (`nn.MultiheadAttention`)** — each position looks at every other position to decide which ones matter for predicting the next character. This is the Q/K/V mechanism from the lecture, just using PyTorch's built-in layer instead of writing the maths manually.
4. **Causal mask** — `torch.triu(..., diagonal=1)` builds a triangular mask that blocks each position from "seeing" future characters, so the model can only use past context when predicting the next one (this is what makes it a *language model* rather than just a text encoder).
5. **Residual connections + LayerNorm** — `self.norm1(h + a)` and `self.norm2(h + self.ffn(h))` add the attention/feed-forward output back onto the input before normalising, exactly the "residuals + LayerNorm" fix for vanishing gradients mentioned in the slides.
6. **Feed-forward network (`self.ffn`)** — a small two-layer MLP applied to every position, matching the "feedforward layer houses most of the processing capacity" point from the slides.
7. **Training loop** — repeatedly samples a random chunk of text, predicts the next character for every position, measures the error with cross-entropy loss, and updates the weights with the AdamW optimizer.
8. **`generate()`** — after training, feeds in a short prompt (`"the cat"`) and repeatedly predicts one new character at a time, sampling from the predicted probability distribution — the same "autoregressive, one token at a time" process described in the Week 3 slides.

## My work

- [notes.md](notes.md) — my notes
- [lab/](lab/) — my lab work for this week
- [reflection.md](reflection.md) — my reflection
