# Week 2 Notes

## Key terms

- **Neuron** — `output = activation(Σ wᵢxᵢ + b)`; weights learned by gradient descent, activation introduces non-linearity (without it, stacked layers collapse into one linear layer).
- **Activation functions** — Sigmoid/Tanh saturate (vanishing gradients); ReLU has a "dying neuron" problem; GELU (BERT/GPT-2) and SwiGLU (LLaMA/PaLM) are the smooth, modern choices used in Transformers.
- **Vanishing gradients** — a sigmoid-chain gradient shrinks by ~0.25 per layer, so 20 layers ≈ 0.5²⁰ ≈ 0.000001; fixed with residual connections + LayerNorm + GELU.
- **Why RNNs fell short** — sequential (can't parallelise), long-range dependencies vanish over steps, single encoder vector is a bottleneck.
- **Query / Key / Value** — Query = "what am I looking for", Key = "what do I represent", Value = "what information do I carry". Every token plays all three roles.
- **Scaled dot-product attention** — `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V`; scaling by `√dₖ` stops the dot products growing too large and saturating softmax.
- **Multi-head attention** — `MultiHead(Q,K,V) = Concat(head₁,...,headₕ)·Wᵒ`; different heads specialise (syntax, coreference, local context, long-range dependencies).

## Important formulas / code

- Attention: `softmax(QKᵀ/√dₖ)V`
- Causal mask in code: `torch.triu(torch.ones(T, T), diagonal=1)` — blocks each position from seeing future tokens.
- GPT-3 uses 96 attention heads (dₖ=128); BERT-base uses 12 heads (dₖ=64).

## Open questions

- Why does scaling specifically by `√dₖ` fix the softmax-saturation problem, rather than some other normalisation?
- Is there a practical way to tell, for a given task, how many attention heads is "enough" before adding more stops helping?
