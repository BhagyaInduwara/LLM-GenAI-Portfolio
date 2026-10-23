# Week 4 Notes

## Key terms

- **PEFT (Parameter-Efficient Fine-Tuning)** — train a small number of extra parameters instead of the whole model; far cheaper and faster.
- **LoRA (Low-Rank Adaptation)** — freezes the base model, injects small trainable low-rank matrices into target layers (here: `q/k/v/o_proj` — the attention projections — and `gate/up/down_proj` — the feed-forward projections).
- **QLoRA** — LoRA on top of a 4-bit quantized base model, for fitting fine-tuning onto smaller GPUs.
- **Chat template** — wraps question/answer pairs in a model's expected conversation format (`<|im_start|>role ... <|im_end|>`) so fine-tuning examples match what the model sees at inference time.
- **SFT (Supervised Fine-Tuning)** — training on input→output pairs; Hugging Face TRL's `SFTTrainer` handles the batching/loss/optimizer loop automatically.
- **Evaluation** — compare base vs. fine-tuned model on held-out questions (not in the training set); BLEU score is one rough, automatic comparison metric.

## Important formulas / code

- LoRA conceptually: `W' = W + B·A`, where `W` is frozen and `B·A` is a much lower-rank, trainable update (rank `r=8` in the lab).
- Trainable parameters in the lab example: ~4.4M out of ~498M total — about 0.88%.
- `LoraConfig(r=8, lora_alpha=16, target_modules=[...])` + `get_peft_model(model, lora_config)`.

## Open questions

- At what point does a training dataset become too small for LoRA fine-tuning to reliably change model behaviour — is there a rough rule of thumb, or does it depend entirely on the base model and domain?
- How does QLoRA's 4-bit quantization actually affect output quality compared to full-precision LoRA, for a model this small?
