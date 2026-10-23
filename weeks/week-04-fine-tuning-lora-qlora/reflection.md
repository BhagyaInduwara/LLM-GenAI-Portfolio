# Week 4 Reflection

## What I learned

Why LoRA is described as "parameter-efficient" in concrete terms, not just as a buzzword: freezing the base model and only training small adapter matrices in the attention and feed-forward projections brought the trainable parameter count down to under 1% of the full model, while still being enough to shift the model's answers toward the fine-tuning domain.

## What was challenging

Checking the lecturer's reference notebook against the lab sheet line by line and finding that its dataset only had 10 question/answer pairs, when the lab sheet explicitly asks for at least 30. It was a useful reminder to actually verify a requirement against the provided example rather than assuming the example is automatically compliant.

## How it connects to earlier weeks

Reuses the exact anatomy from Week 3 — tokenizer, embeddings, Transformer blocks, LM head — except now fine-tuning a real pretrained model (`Qwen2.5-0.5B-Instruct`) instead of training a tiny model from scratch. The `q/k/v/o_proj` layers LoRA targets are literally the Query/Key/Value/Output projections from Week 2's attention mechanics.

## Questions I still have

Whether 30 Q&A pairs (the lab sheet's minimum) is actually enough to see a clear, consistent behaviour shift, or whether that's really just enough to demonstrate the fine-tuning *mechanics* rather than get a genuinely reliable model.

## What I'd try next

Run `lab/fine_tuning_lora_qlora_complete.ipynb` on a GPU and compare the base vs. fine-tuned answers properly, then try doubling the dataset size to see whether more examples visibly improves how consistently the fine-tuned model sticks to the shop's actual details.
