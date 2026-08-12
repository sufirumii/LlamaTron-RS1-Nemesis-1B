# LlamaTron RS1 Nemesis

A 1B-parameter medical reasoning model, fine-tuned with QLoRA on 204K clinical reasoning conversations — trained end-to-end in under 4 hours on a single GPU.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38A169?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-Llama%203.2%201B%20Instruct-4A5568?style=flat-square)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-204K%20samples-2B6CB0?style=flat-square)](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1)
[![Method](https://img.shields.io/badge/Method-QLoRA-7C3AED?style=flat-square)](https://github.com/sufirumii/LlamaTron-RS1-Nemesis-1B)

## Overview

LlamaTron RS1 Nemesis is a compact medical reasoning model built by fine-tuning `meta-llama/Llama-3.2-1B-Instruct` with QLoRA on the `Medical-Reasoning-SFT-MiniMax-M2.1` dataset — 204,773 clinical reasoning conversations with full chain-of-thought traces covering differential diagnosis, treatment planning, pharmacology, and clinical case analysis.

Despite being a 1-billion-parameter model, LlamaTron RS1 Nemesis handles complex clinical questions with structured, coherent reasoning, trained in under 4 hours on a single NVIDIA H200 GPU.

## Architecture


<img width="1800" height="1160" alt="image" src="https://github.com/user-attachments/assets/cdc51662-1eef-4d45-8317-4b5f4aeda232" />


Only 5.6M of the model's 1.24B parameters are updated during training (0.45%) — the base weights stay frozen and quantized to 4-bit throughout, with the LoRA adapters carrying all of the learned clinical-reasoning behavior. This is what keeps a full fine-tuning run on 204K samples under 4 hours on a single GPU.

## Demo

**Interface**

<img width="1451" height="498" alt="Interface preview" src="https://github.com/user-attachments/assets/f5cb097e-b340-44d2-baf4-ee29a4ee4267" />

**Model response example**

<img width="1444" height="758" alt="Model response example" src="https://github.com/user-attachments/assets/fdb45f6f-f983-4326-a9de-f792b6d7bdd5" />

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | meta-llama/Llama-3.2-1B-Instruct |
| GPU | NVIDIA H200 |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank | r=8, alpha=16 |
| Trainable parameters | 5.6M / 1.24B (0.45%) |
| Effective batch size | 32 (8 x 4 accumulation steps) |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | paged_adamw_8bit |
| Max sequence length | 512 |
| Epochs | 1 |
| Training time | 3 hours 59 minutes |
| Total steps | 6,271 |

## Training Results


<img width="1600" height="840" alt="training-loss" src="https://github.com/user-attachments/assets/77ffac5b-0826-4c50-a3f3-1d4b53e159de" />


| Step | Train Loss | Val Loss |
|------|------------|----------|
| 500 | 1.5759 | 1.6126 |
| 1000 | 1.5176 | 1.5538 |
| 2000 | 1.4795 | 1.5060 |
| 3000 | 1.4534 | 1.4814 |
| 4000 | 1.4228 | 1.4662 |
| 5000 | 1.4301 | 1.4567 |
| 6271 | 1.4200 | 1.4500 |

Loss decreased consistently across all 6,271 steps, with clean convergence and no overfitting observed — validation loss tracks training loss closely throughout rather than diverging.

## Dataset

Trained on [Medical-Reasoning-SFT-MiniMax-M2.1](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1) by [Maziyar Panahi](https://www.linkedin.com/in/maziyar-panahi/) (OpenMed).

| Property | Value |
|----------|-------|
| Total samples | 204,773 |
| Estimated tokens | ~621 million |
| Format | Multi-turn chat with chain-of-thought reasoning |
| License | Apache 2.0 |
| Topics | Differential diagnosis, treatment planning, pharmacology, clinical case analysis |

## Usage

The model is hosted on the Hugging Face Hub as `Rumiii/LlamaTron-RS1-Nemesis-1B`. Both loading methods below pull directly from the Hub — no local files required.

**Quick start with `pipeline`:**

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="Rumiii/LlamaTron-RS1-Nemesis-1B")

messages = [
    {"role": "system", "content": "You are LlamaTron RS1 Nemesis, a knowledgeable and compassionate medical AI assistant."},
    {"role": "user", "content": "What are the symptoms of Type 2 Diabetes?"},
]

output = pipe(messages, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9)
print(output[0]["generated_text"][-1]["content"])
```

**Manual loading with the chat template:**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Rumiii/LlamaTron-RS1-Nemesis-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {"role": "system", "content": "You are LlamaTron RS1 Nemesis, a knowledgeable and compassionate medical AI assistant."},
    {"role": "user", "content": "What are the symptoms of Type 2 Diabetes?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Requirements

```
torch>=2.0.0
transformers>=4.44.0
peft>=0.12.0
trl>=0.9.6
bitsandbytes>=0.43.3
accelerate>=0.33.0
datasets>=2.21.0
```

## Limitations

- Not evaluated against a clinical benchmark (e.g. MedQA, USMLE-style item banks) — the loss curves show clean convergence, not clinical accuracy
- At 1B parameters and rank-8 LoRA, the model has materially less capacity than larger medical LLMs and is more prone to confident but incorrect statements
- Trained on English-language conversations only
- One training epoch over the dataset — samples were seen once, which limits how deeply any single reasoning pattern was reinforced

## Disclaimer

LlamaTron RS1 Nemesis is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

## Credits

- **Dataset**: [Maziyar Panahi](https://www.linkedin.com/in/maziyar-panahi/), founder of [OpenMed](https://huggingface.co/OpenMed), for releasing the `Medical-Reasoning-SFT-MiniMax-M2.1` dataset openly under Apache 2.0
- **Base model**: Meta AI for releasing `Llama-3.2-1B-Instruct`
- **Libraries**: Hugging Face, PEFT, TRL, BitsAndBytes

## Author

Rumi Iqbal Sufi
AI Engineer
GitHub: https://github.com/sufirumii
Hugging Face: https://huggingface.co/Rumiii
LinkedIn: https://www.linkedin.com/in/rumi-sufi-6323a5265/

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
