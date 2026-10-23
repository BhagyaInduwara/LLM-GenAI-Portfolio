# Week 3 Reflection

## What I learned

How the pieces from Weeks 1–2 (tokenizer, embeddings, attention) chain together into a full generative model: tokenizer → Transformer blocks → LM head → probability distribution → sample one token → repeat. Seeing that "autoregressive generation" is really just that loop running over and over made LLM text generation feel far less like a black box.

## What was challenging

Spotting that the lecturer's reference notebook (`resources/tiny_transformer_llm.ipynb`) was actually broken — its Gradio chat cell calls a `generate()` function that is never defined anywhere in the notebook, and the lab sheet's Step 9 ("generate stories from a prompt") is missing entirely. Diagnosing *why* it was broken (not just that it would error) took carefully tracing which lab sheet step each code cell was supposed to correspond to.

## How it connects to earlier weeks

Directly extends Week 2's attention mechanics into a complete generative model, and reuses Week 1's tokenization idea — just at the character level instead of word/sub-word level, which made the "text as data" concept from Week 1 concrete again in a new context.

## Questions I still have

Now that I've seen the gap in the reference notebook, I'm not sure how common bugs like this are in lecture materials generally — is comparing provided code against the written task list (like I did here) something worth doing as a habit for every lab, or was this one unusual?

## What I'd try next

Actually run `lab/tiny_transformer_llm_complete.ipynb` on a GPU, then try the Step 10 experiment properly — train the 4-layer variant alongside the original 2-layer one and compare both the loss curves and the generated text quality.
