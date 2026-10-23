# Week 4 — Fine-Tuning LLMs with LoRA/QLoRA

**Module:** PUSL3205 — Large Language Models and Generative AI
**Topic:** Parameter-efficient fine-tuning of a small instruction-tuned LLM

## Key concepts covered

- **Why fine-tune at all** — adapting a general-purpose pretrained LLM to answer questions about a specific, narrow domain (e.g. a shop's own products/policies) rather than prompting alone.
- **Parameter-Efficient Fine-Tuning (PEFT)** — instead of updating all of a model's weights, only train a small number of extra parameters, which is far cheaper and faster.
- **LoRA (Low-Rank Adaptation)** — freezes the original model weights and injects small trainable "adapter" matrices into specific layers (attention projections, feed-forward projections); here only ~0.9% of the model's parameters are actually trained.
- **QLoRA** — the same idea as LoRA, but the frozen base model is loaded in 4-bit quantized form first, so it fits on smaller/cheaper GPUs.
- **Chat templates** — formatting question/answer pairs into the model's expected conversation format (`<|im_start|>user ... <|im_end|>`) so it fine-tunes correctly for instruction-following.
- **Supervised fine-tuning (SFT)** — training the model on input→output pairs using Hugging Face TRL's `SFTTrainer`.
- **Evaluating a fine-tune** — comparing the base model's answers against the fine-tuned model's answers on the same held-out questions, including BLEU score as one (limited) quantitative measure.

## Resources (from lecturer)

- [Lab sheet 4.pdf](resources/Lab%20sheet%204.pdf) — Lab sheet: fine-tune a small LLM with LoRA/QLoRA.
- [fine_tune_generative_models.ipynb](resources/fine_tune_generative_models.ipynb) — Lecturer-provided reference notebook for the lab.

## What the Week 4 lab involves

1. Install `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`.
2. Check what GPU is available and why it matters for fine-tuning.
3. Pick a small instruction-tuned model that fits the available GPU (e.g. `Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM2-360M-Instruct`).
4. Build a small dataset of at least 30 question/answer pairs about one specific domain (the example notebook uses a fictional university shop, "Campus Essentials").
5. Configure LoRA (or QLoRA with 4-bit quantization if GPU memory is limited) using the PEFT library.
6. Fine-tune with a supervised fine-tuning workflow (TRL's `SFTTrainer`).
7. Test the original model and the fine-tuned model on the same questions, including questions **not** in the training set, and compare the answers.

## Explaining the lecturer's code: `fine_tune_generative_models.ipynb`

1. **Install libraries & check the GPU** — installs `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`, then checks `torch.cuda.is_available()` and prints the GPU name/memory (a Tesla T4 with ~14.6 GB in this run). A GPU is needed because fine-tuning means repeated forward+backward passes through a model with hundreds of millions of parameters — far too slow on CPU.
2. **Build the dataset** — a small Python list of `{"question": ..., "answer": ...}` dicts about a fictional shop ("Campus Essentials"), turned into a Hugging Face `Dataset` with `Dataset.from_list(data)`.
3. **Load the base model** — `AutoTokenizer`/`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`. This is the same "tokenizer + stack of Transformer blocks + LM head" structure from Week 3, just a real pretrained 0.5-billion-parameter model instead of a tiny from-scratch one.
4. **Format with a chat template** — `tokenizer.apply_chat_template(...)` wraps each question/answer pair in the model's expected `<|im_start|>role ... <|im_end|>` format, so the model is fine-tuned on examples that look exactly like the conversations it will see at inference time.
5. **Configure LoRA** — `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], ...)` targets the attention projection layers (`q/k/v/o_proj` — directly the Query/Key/Value/Output matrices from the Week 2 attention lecture) and the feed-forward layers (`gate/up/down_proj`), adding small trainable low-rank matrices to each. `get_peft_model(model, lora_config)` then freezes everything else — the printout shows only ~0.88% of all parameters are trainable.
6. **Train with `SFTTrainer`** — `SFTConfig` sets training hyperparameters (3 epochs, small batch size, learning rate `2e-4`, FP16 precision), and `SFTTrainer(...).train()` runs the actual training loop (forward pass → loss → backward pass → weight update), the same cycle described in Week 2's backpropagation slides, just handled automatically by the TRL library instead of a hand-written loop.
7. **Save the adapter** — `trainer.model.save_pretrained(output_dir)` saves only the small LoRA adapter weights (a few MB), not a full copy of the base model — one of the main benefits of PEFT.
8. **Compare base vs. fine-tuned** — reloads the base model fresh, loads the fine-tuned version with `PeftModel.from_pretrained(base_model, output_dir)`, and asks both the same held-out questions (not seen during training) via a shared `generate_answer()` helper, printing both answers side by side in a pandas DataFrame.
9. **BLEU score** — `sacrebleu.corpus_bleu(...)` scores each model's answers against reference answers as one rough, automatic way to compare output quality (low scores here reflect how little data — only 10–30 examples — the model was fine-tuned on).

## My work

- [notes.md](notes.md) — my notes
- [lab/fine_tuning_lora_qlora_complete.ipynb](lab/fine_tuning_lora_qlora_complete.ipynb) — my completed version of the lab, built against Lab sheet 4's checklist: all 8 steps covered, including the GPU question answered in a markdown cell and the dataset expanded to the required 30 question/answer pairs (the lecturer's reference notebook only had 10). Not yet run — open it in Colab with a GPU runtime and run it top to bottom.
- [reflection.md](reflection.md) — my reflection
