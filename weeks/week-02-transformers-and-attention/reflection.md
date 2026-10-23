# Week 2 Reflection

## What I learned

How self-attention actually works mechanically — Query/Key/Value as three roles every token plays, scaled dot-product attention as a weighted lookup, and multi-head attention as several of those lookups running in parallel so the model can track different relationships (syntax, coreference, long-range context) at once. Tracing the neuron → MLP → backpropagation → vanishing-gradient chain first made it clear *why* Transformers needed residual connections and LayerNorm, rather than those just being arbitrary design choices.

## What was challenging

The attention maths took a few passes to click, especially *why* the scaling factor `√dₖ` matters (it stops the softmax from saturating on large dot products) rather than it just being a cosmetic detail. Working through `transformer_practice.ipynb` line by line — matching each part of the code (causal mask, residual connections, feed-forward block) back to the slide it came from — was the thing that actually made it stick.

## How it connects to earlier weeks

Builds directly on Week 1: the embeddings from Word2Vec/BERT are exactly the vectors that Query, Key, and Value are computed from, and the Week 1 "neural architectures" comparison slide (RNN/LSTM/CNN/Transformer) is what this week unpacked into full detail for the Transformer.

## Questions I still have

In the group project (PackWise), we used an LLM as a brainstorming/product-development partner rather than writing attention code — I'd like a clearer picture of where the line is between "using an LLM as a tool" in a project like that and actually needing to understand the attention mechanics underneath it.

## What I'd try next

Run `transformer_practice.ipynb` with a different number of attention heads and compare the generated text quality, to get a feel for what multi-head attention is actually buying over single-head attention on a small dataset.
