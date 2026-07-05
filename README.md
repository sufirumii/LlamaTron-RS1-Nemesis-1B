# LlamaTron RS1 Nemesis

> **Base Model:** `meta-llama/Llama-3.2-1B-Instruct` fine-tuned on `OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1`

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model: Llama 3.2 1B](https://img.shields.io/badge/Base%20Model-Llama%203.2%201B-orange)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Dataset: OpenMed](https://img.shields.io/badge/Dataset-OpenMed-green)](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1)
[![Method: QLoRA](https://img.shields.io/badge/Method-QLoRA-purple)](https://github.com/sufirumii/LlamaTron-RS1-Nemesis-1B)

---

## Overview

**LlamaTron RS1 Nemesis** is a compact medical reasoning model built by fine-tuning `meta-llama/Llama-3.2-1B-Instruct` using QLoRA on the `Medical-Reasoning-SFT-MiniMax-M2.1` dataset — 204,773 clinical reasoning conversations with full chain-of-thought traces covering differential diagnosis, treatment planning, pharmacology, and clinical case analysis.

Despite being a 1-billion-parameter model, LlamaTron RS1 Nemesis handles complex clinical questions with structured, coherent reasoning, trained in under 4 hours on a single NVIDIA H200 GPU.

---

## Demo

**Interface**

<img width="1451" height="498" alt="Interface preview" src="https://github.com/user-attachments/assets/f5cb097e-b340-44d2-baf4-ee29a4ee4267" />

**Model response example**

<img width="1444" height="758" alt="Model response example" src="https://github.com/user-attachments/assets/fdb45f6f-f983-4326-a9de-f792b6d7bdd5" />

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-3.2-1B-Instruct |
| GPU | NVIDIA H200 |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank | r=8, alpha=16 |
| Trainable Parameters | 5.6M / 1.24B (0.45%) |
| Effective Batch Size | 32 (8 x 4 accumulation steps) |
| Learning Rate | 2e-4 (cosine schedule) |
| Optimizer | paged_adamw_8bit |
| Max Sequence Length | 512 |
| Epochs | 1 |
| Training Time | 3 hours 59 minutes |
| Total Steps | 6,271 |

## Training Results

| Step | Train Loss | Val Loss |
|------|------------|----------|
| 500 | 1.5759 | 1.6126 |
| 1000 | 1.5176 | 1.5538 |
| 2000 | 1.4795 | 1.5060 |
| 3000 | 1.4534 | 1.4814 |
| 4000 | 1.4228 | 1.4662 |
| 5000 | 1.4301 | 1.4567 |
| 6271 | 1.4200 | 1.4500 |

Loss decreased consistently across all 6,271 steps, with clean convergence and no overfitting observed.

---

## Dataset

Trained on [Medical-Reasoning-SFT-MiniMax-M2.1](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1) by [Maziyar Panahi](https://www.linkedin.com/in/maziyar-panahi/) (OpenMed).

| Property | Value |
|----------|-------|
| Total Samples | 204,773 |
| Estimated Tokens | ~621 Million |
| Format | Multi-turn chat with chain-of-thought reasoning |
| License | Apache 2.0 |
| Topics | Differential diagnosis, treatment planning, pharmacology, clinical case analysis |

---

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

---

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

---

## Disclaimer

LlamaTron RS1 Nemesis is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## Credits

- **Dataset**: [Maziyar Panahi](https://www.linkedin.com/in/maziyar-panahi/), founder of [OpenMed](https://huggingface.co/OpenMed), for releasing the `Medical-Reasoning-SFT-MiniMax-M2.1` dataset openly under Apache 2.0
- **Base Model**: Meta AI for releasing `Llama-3.2-1B-Instruct`
- **Libraries**: Hugging Face, PEFT, TRL, BitsAndBytes

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
