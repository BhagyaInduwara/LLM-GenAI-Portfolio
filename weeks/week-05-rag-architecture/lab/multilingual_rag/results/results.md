## Retrieval results (Hit@1 / Hit@4 / MRR)

| Embedding model | Strategy | English | Sinhala | Tamil |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (English-only) | direct | 0.83 / 0.92 / 0.87 | 0.25 / 0.50 / 0.40 | 0.33 / 0.42 / 0.44 |
| all-MiniLM-L6-v2 (English-only) | translated | 0.83 / 0.92 / 0.87 | 0.17 / 0.75 / 0.43 | 0.75 / 0.92 / 0.82 |
| multilingual-e5-small | direct | 0.92 / 0.92 / 0.93 | 0.67 / 0.92 / 0.79 | 0.67 / 0.83 / 0.78 |
| multilingual-e5-small | translated | 0.92 / 0.92 / 0.93 | 0.17 / 0.58 / 0.36 | 0.92 / 0.92 / 0.93 |
| qwen3-embedding:0.6b (Ollama) | direct | 0.83 / 0.92 / 0.89 | 0.17 / 0.75 / 0.39 | 0.25 / 0.67 / 0.51 |
| qwen3-embedding:0.6b (Ollama) | translated | 0.83 / 0.92 / 0.89 | 0.17 / 0.58 / 0.39 | 0.83 / 0.92 / 0.89 |

## LLM translations of the questions

| # | Sinhala → English | Tamil → English |
|---|---|---|
| 1 | What is the cause of the sudden increase in the number of accidents? | How many percent of attendance is required to pass the final exam? |
| 2 | What is the earliest date of the death of Ashoka? | What is the time frame for applying for higher education? |
| 3 | Do you have a book to read before bedtime? | How many books can be borrowed from the library for how long? |
| 4 | What is the cost of using AI in daily life? | What penalty would there be if AI development usage was not announced? |
| 5 | Do you have any experience with ChatGPT? | Can I use ChatGPT to edit my draft? |
| 6 | How many types of lachanakiri are there in the Viharaya? | How many credits will be awarded for the final year research project? |
| 7 | What is the name of the most beautiful flower in the village of Thoragaya? | What research topics can I choose for my final year project? |
| 8 | Do you have any idea how much it costs to rent a room? | What is the minimum score required to pass the main subject? |
| 9 | What is the name of the capital of the Pariyas? | How long is a Bachelor's degree in Computer Science? |
| 10 | Do you have a coconut tree in your village? | What items are prohibited in the examination hall? |
| 11 | What is the price of the book "Pusthakasala" by Premasara? | How much punishment would be given if a book from the library is returned late? |
| 12 | Do you have a camera to take pictures of the Buddha statues? | How many days before can I pre-register for the group discussion room? |

## Misses (answer chunk not in top 4)

| Model | Strategy | Lang | Q# | Rank of answer chunk |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (English-only) | direct | en | 7 | 7 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 3 | 5 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 6 | 6 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 7 | 8 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 9 | 11 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 11 | 5 |
| all-MiniLM-L6-v2 (English-only) | direct | si | 12 | 7 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 1 | 7 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 3 | 9 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 6 | 5 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 7 | 10 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 9 | 8 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 10 | 8 |
| all-MiniLM-L6-v2 (English-only) | direct | ta | 12 | 9 |
| all-MiniLM-L6-v2 (English-only) | translated | en | 7 | 7 |
| all-MiniLM-L6-v2 (English-only) | translated | si | 4 | 8 |
| all-MiniLM-L6-v2 (English-only) | translated | si | 10 | 12 |
| all-MiniLM-L6-v2 (English-only) | translated | si | 12 | 7 |
| all-MiniLM-L6-v2 (English-only) | translated | ta | 7 | 7 |
| multilingual-e5-small | direct | en | 7 | 5 |
| multilingual-e5-small | direct | si | 7 | 7 |
| multilingual-e5-small | direct | ta | 1 | 5 |
| multilingual-e5-small | direct | ta | 7 | 9 |
| multilingual-e5-small | translated | en | 7 | 5 |
| multilingual-e5-small | translated | si | 1 | 12 |
| multilingual-e5-small | translated | si | 2 | 7 |
| multilingual-e5-small | translated | si | 4 | 9 |
| multilingual-e5-small | translated | si | 9 | 9 |
| multilingual-e5-small | translated | si | 10 | 12 |
| multilingual-e5-small | translated | ta | 7 | 5 |
| qwen3-embedding:0.6b (Ollama) | direct | en | 7 | 6 |
| qwen3-embedding:0.6b (Ollama) | direct | si | 4 | 5 |
| qwen3-embedding:0.6b (Ollama) | direct | si | 9 | 5 |
| qwen3-embedding:0.6b (Ollama) | direct | si | 10 | 6 |
| qwen3-embedding:0.6b (Ollama) | direct | ta | 2 | 6 |
| qwen3-embedding:0.6b (Ollama) | direct | ta | 4 | 5 |
| qwen3-embedding:0.6b (Ollama) | direct | ta | 7 | 5 |
| qwen3-embedding:0.6b (Ollama) | direct | ta | 8 | 5 |
| qwen3-embedding:0.6b (Ollama) | translated | en | 7 | 6 |
| qwen3-embedding:0.6b (Ollama) | translated | si | 2 | 5 |
| qwen3-embedding:0.6b (Ollama) | translated | si | 4 | 5 |
| qwen3-embedding:0.6b (Ollama) | translated | si | 8 | 8 |
| qwen3-embedding:0.6b (Ollama) | translated | si | 9 | 8 |
| qwen3-embedding:0.6b (Ollama) | translated | si | 10 | 6 |
| qwen3-embedding:0.6b (Ollama) | translated | ta | 7 | 6 |
